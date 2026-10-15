"""Derived-dataset store on the T9 SSD.

The internal disk has ~32 GB free and the L1 layer alone is a few GB, so every
generated artifact larger than a report lives on the external drive. Layout::

    /Volumes/T9/nq_levels/
        bars_1s/    YYYY-MM.parquet     L1  1-second bars
        profiles/   YYYY-MM.parquet     L1  volume-at-price, 30-min buckets
        daily/      daily.parquet       L2  one row per (session, symbol)
        candidates/ ...                 L3  candidate zones at the decision cut
        labels/     ...                 L4  touch + reaction outcomes
        micro/      ...                 L5  MBP-1 reductions

Monthly partitions are the unit of resumability: a failed or interrupted build
re-runs one month (~21 sessions, tens of seconds) rather than five years. Every
write goes through :func:`write_partition`, which refuses paths under a raw
root and writes atomically via a temp file, so an interrupted run can never
leave a half-written partition that a later read would silently treat as
complete.
"""
from __future__ import annotations

import os
import shutil
from datetime import date
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import yaml

from .raw import assert_not_raw, load_paths


def derived_root(configs_dir: str) -> Path:
    return Path(load_paths(configs_dir)["derived"]["root"])


def layer_dir(layer: str, configs_dir: str) -> Path:
    d = load_paths(configs_dir)["derived"]
    root = Path(d["root"])
    if layer not in d:
        raise KeyError(f"unknown derived layer {layer!r}; known: "
                       f"{sorted(k for k in d if k != 'root')}")
    p = root / d[layer]
    p.mkdir(parents=True, exist_ok=True)
    return p


def month_key(d: date) -> str:
    return f"{d.year:04d}-{d.month:02d}"


def partition_path(layer: str, key: str, configs_dir: str) -> Path:
    return layer_dir(layer, configs_dir) / f"{key}.parquet"


def write_partition(df: pd.DataFrame, layer: str, key: str, configs_dir: str,
                    compression: str = "zstd") -> Path:
    """Atomically write one partition. Never lands a partial file."""
    path = assert_not_raw(partition_path(layer, key, configs_dir))
    tmp = path.with_suffix(".parquet.tmp")
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), tmp,
                   compression=compression)
    os.replace(tmp, path)
    return path


def partition_exists(layer: str, key: str, configs_dir: str) -> bool:
    return partition_path(layer, key, configs_dir).exists()


def _parquets(d: Path) -> List[Path]:
    """Parquet files in a directory, excluding macOS AppleDouble sidecars.

    exFAT has no resource forks, so macOS writes `._name` companions next to
    every file. They match `*.parquet` and pyarrow fails on them with a
    "magic bytes not found" error, so they must be filtered at every read.
    """
    return sorted(p for p in d.glob("*.parquet") if not p.name.startswith("._"))


def list_partitions(layer: str, configs_dir: str) -> List[str]:
    return [p.stem for p in _parquets(layer_dir(layer, configs_dir))]


def read_layer(layer: str, configs_dir: str,
               sessions: Optional[Sequence[date]] = None,
               months: Optional[Sequence[str]] = None,
               columns: Optional[Sequence[str]] = None,
               filters=None) -> pd.DataFrame:
    """Read a derived layer, optionally restricted to months and/or sessions.

    Restricting by ``months`` is what keeps experiments cheap: reading two
    months of 1-second bars costs a few hundred MB instead of the whole layer.
    """
    d = layer_dir(layer, configs_dir)
    if months is None and sessions is not None:
        months = sorted({month_key(s) for s in sessions})
    files = ([d / f"{m}.parquet" for m in months] if months is not None
             else _parquets(d))
    files = [f for f in files if f.exists() and not f.name.startswith("._")]
    if not files:
        return pd.DataFrame()
    tbl = ds.dataset([str(f) for f in files]).to_table(
        columns=list(columns) if columns else None, filter=filters)
    df = tbl.to_pandas()
    if sessions is not None and "session" in df.columns:
        want = {pd.Timestamp(s).date() for s in sessions}
        df = df[df["session"].isin(want)]
    return df.reset_index(drop=True)


def write_report(obj, name: str, project_root: Path) -> Path:
    """Small JSON/markdown artifacts stay in the repo, not on the SSD."""
    out = assert_not_raw(Path(project_root) / "outputs" / name)
    out.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(obj, str):
        out.write_text(obj)
    else:
        import json
        out.write_text(json.dumps(obj, indent=2, default=str))
    return out
