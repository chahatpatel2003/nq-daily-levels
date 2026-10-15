"""Snapshot assembly + JSON schema v1 + text rendering."""
from __future__ import annotations

import json
from datetime import datetime, timezone

from .regions import build_regions

SCHEMA_VERSION = 1


def render(engine, session: str, at_sec: int, max_distance: float = None,
           untouched_only: bool = False) -> dict:
    snap = engine.snapshot(session, at_sec)
    px = snap["price"]
    regions = build_regions(snap["zones"])
    if max_distance is not None:
        regions = [r for r in regions
                   if (r["lo"] - px) / 4 <= max_distance]
    if untouched_only:
        regions = [r for r in regions if r["n_untouched"] > 0]
    out = dict(schema_version=SCHEMA_VERSION,
               generated_utc=datetime.now(timezone.utc).isoformat(),
               session=session, at_sec=at_sec, symbol="NQ (front, adj)",
               price=px / 4,
               n_zones_above=len(snap["zones"]),
               n_regions_above=len(regions),
               regions=[_region_json(r, px) for r in regions])
    return out


def _region_json(r, px):
    return dict(lo=r["lo"] / 4, hi=r["hi"] / 4,
                distance_pts=(r["lo"] - px) / 4,
                width_pts=r["width_pts"], n_children=r["n_children"],
                n_pivots=r["n_pivots"], scales=r["scales"],
                max_scale=r["max_scale"], n_untouched=r["n_untouched"],
                max_depth=r["max_depth"],
                core2=[[a / 4, b / 4] for a, b in r["core2"]],
                children=[dict(zid=int(c["zid"]), lo=c["lo"] / 4,
                               hi=c["hi"] / 4, scale=int(c["scale"]),
                               taps=int(c["n_taps"]),
                               age_sessions=int(c["age_sessions"]))
                          for c in r["children"]])


def to_text(js) -> str:
    L = [f"NQ Resistance Engine  (schema v{js['schema_version']})",
         f"Session: {js['session']}  t+{js['at_sec']}s",
         f"Price: {js['price']:.2f}  (roll-adjusted front)",
         f"Zones above: {js['n_zones_above']}  Regions shown: "
         f"{js['n_regions_above']}", ""]
    for i, r in enumerate(js["regions"][:6], 1):
        L.append(f"REGION {i}: {r['lo']:.2f} - {r['hi']:.2f}  "
                 f"(+{r['distance_pts']:.1f} pts, {r['width_pts']:.1f} wide)")
        L.append(f"  children {r['n_children']} | pivots {r['n_pivots']} | "
                 f"scales {','.join('S'+str(s) for s in r['scales'])} | "
                 f"untouched {r['n_untouched']} | depth {r['max_depth']}")
        for a, b in r["core2"]:
            L.append(f"  CORE-2: {a:.2f} - {b:.2f}")
        for c in r["children"][:8]:
            st = "untouched" if c["taps"] == 0 else f"tap{c['taps']}"
            L.append(f"    Z{c['zid']:<6} {c['lo']:.2f}-{c['hi']:.2f}  "
                     f"scale{c['scale']}  {st}  age{c['age_sessions']}")
        L.append("")
    return "\n".join(L)


def save(js, path_json, path_txt=None):
    with open(path_json, "w") as f:
        json.dump(js, f, indent=1)
    if path_txt:
        with open(path_txt, "w") as f:
            f.write(to_text(js))
