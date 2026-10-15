"""Read-only Phase-22 region presentation layer.

regions_of / depth_profile / core_bands are IMPORTED from the audited
P22 module (scripts/run_p22_regions.py) — not re-implemented. Regions
never mutate child zones; bounds are provably min/max of children
(asserted inside regions_of)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
from run_p22_regions import regions_of, depth_profile, core_bands  # noqa: E402


def build_regions(zones_above: pd.DataFrame) -> list:
    """zones_above: snapshot['zones'] (sorted by lo). Returns region
    dicts with child indices, cores, composition."""
    if not len(zones_above):
        return []
    L = zones_above["lo"].to_numpy()
    H = zones_above["hi"].to_numpy()
    regs = regions_of(L, H, None)
    out = []
    for r in regs:
        mem = r["members"]
        sub = zones_above.iloc[mem]
        edges, depth = depth_profile(L[mem], H[mem])
        c2 = core_bands(edges, depth, 2)
        c3 = core_bands(edges, depth, 3)
        # core interval list (depth>=2) for display
        cores2 = []
        if len(depth):
            run_s = None
            for i in range(len(depth)):
                if depth[i] >= 2 and run_s is None:
                    run_s = edges[i]
                if (depth[i] < 2 or i == len(depth) - 1) and run_s is not None:
                    run_e = edges[i] if depth[i] < 2 else edges[i + 1]
                    cores2.append((float(run_s), float(run_e)))
                    run_s = None
        piv = set((s, int(fm // 10)) for s, fm in
                  zip(sub["session"], sub["form_min"]))
        out.append(dict(
            lo=float(r["lo"]), hi=float(r["hi"]),
            width_pts=float((r["hi"] - r["lo"]) / 4),
            n_children=len(mem),
            n_pivots=len(piv),
            scales=sorted(set(int(x) for x in sub["scale"])),
            max_scale=int(sub["scale"].max()),
            n_untouched=int((sub["n_taps"] == 0).sum()),
            max_depth=int(depth.max()) if len(depth) else 1,
            core2=cores2,
            core2_longest_pts=float(c2[1] / 4),
            core3_present=bool(c3[2] > 0),
            children=sub[["zid", "lo", "hi", "scale", "n_taps",
                          "age_sessions", "session"]]
            .to_dict("records")))
    return out
