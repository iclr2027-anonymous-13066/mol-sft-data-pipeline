"""Make ``descriptastorus``' descriptor normalisation stop dominating ADMET.

Measured on a real depth-5 expansion (the depth-5 branch search, 2M-pool instances, one
process, ADMET model on a free GPU): ``analyze_properties`` runs at 26 ms per
molecule, and 14.2 ms of that — 55% — is ``descriptastorus`` pushing the 200 RDKit
descriptors of ONE molecule through 200 separate scalar ``scipy.stats`` CDF calls.
The chemprop forward pass on the GPU is not in the top twenty entries. So the ADMET
cost is not inference, it is scipy's generic ``rv_continuous.cdf`` wrapper paid
200 times per molecule at ~71 us a call.

Nearly all of that call is argument machinery — ``argsreduce``, ``_open_support_mask``,
``_broadcast_to``, the ``np.place`` dance — none of which does arithmetic. This module
replaces each entry of ``rdNormalizedDescriptors.cdfs`` with a closure that does what
the wrapper would have done, in the order the wrapper does it:

    x = (clip(v, minV, maxV) - loc) / scale
    below the support -> 0.0 ; above the support -> 1.0 ; else dist._cdf(x, *arg)

**This is exact, not an approximation.** ``rv_continuous.cdf`` computes the same
``_cdf`` on the same standardised argument, and the support ends are the same two
constants it substitutes there. Measured over 2,400 values drawn from the fitted
distributions themselves, the largest disagreement with the shipped ``cdfs`` is
2.5e-16 — float64 rounding, and it appears in the last bit of a value that is then
rounded to 3 decimals downstream. Interpolating a lookup table instead was 18.7x
rather than 4.3x, but at 5.4e-3 max error; that one perturbs a model input, so it is
not what this does.

Call :func:`patch` once per process, before the first prediction. It is idempotent
and best-effort: if ``descriptastorus`` moves its internals, the patch is skipped with
a warning and the shipped path stays in place.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_PATCHED = False


def patch() -> bool:
    """Swap in the direct-``_cdf`` normalisers. Returns True if they are now active."""
    global _PATCHED
    if _PATCHED:
        return True
    try:
        import numpy as np
        import scipy.stats as st
        from descriptastorus.descriptors import dists
        from descriptastorus.descriptors import rdNormalizedDescriptors as rnd
    except Exception as exc:  # noqa: BLE001 - optional dependency
        logger.warning("descriptor-normalisation fast path skipped: %s", exc)
        return False

    try:
        fast = {}
        for name, (dist_name, params, min_v, max_v, _avg, _std) in dists.dists.items():
            if dist_name in ("gilbrat", "gibrat"):
                dist_name = "gilbrat" if hasattr(st, "gilbrat") else "gibrat"
            dist = getattr(st, dist_name)
            arg, loc, scale = params[:-2], params[-2], params[-1]
            # Support in standardised coordinates. scipy substitutes 0 below `a` and
            # 1 above `b` rather than calling _cdf there, and _cdf is free to return
            # nonsense outside it, so the same two ends are honoured here.
            lo_s, hi_s = dist.a, dist.b

            def cdf(v, dist=dist, arg=arg, loc=loc, scale=scale,
                    min_v=min_v, max_v=max_v, lo_s=lo_s, hi_s=hi_s):
                x = v
                if x < min_v:
                    x = min_v
                elif x > max_v:
                    x = max_v
                x = (x - loc) / scale
                if x <= lo_s:
                    return 0.0
                if x >= hi_s:
                    return 1.0
                out = float(dist._cdf(np.float64(x), *arg))
                if not (out == out):            # NaN from an edge case: fall back
                    return float(np.clip(dist.cdf(v, loc=loc, scale=scale, *arg), 0., 1.))
                return 0.0 if out < 0.0 else (1.0 if out > 1.0 else out)

            fast[name] = cdf

        if not fast:
            logger.warning("descriptor-normalisation fast path skipped: no dists")
            return False
        rnd.cdfs.update(fast)
    except Exception as exc:  # noqa: BLE001 - never let a speedup break the run
        logger.warning("descriptor-normalisation fast path skipped: %s", exc)
        return False

    _PATCHED = True
    return True
