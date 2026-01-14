import argparse
import io
import json
import math
import os
import re
import tempfile
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class RankAverages:
    rank: int
    forward_ms: float
    backward_ms: float


@dataclass(frozen=True)
class RankTimelineSegment:
    rank: int
    start: float
    duration: float
    phase: str
    iter: int
    mb: int


@dataclass
class LiveRenderState:
    timeline_png: Optional[bytes] = None
    heatmap_png: Optional[bytes] = None
    last_update_s: float = 0.0
    last_error: Optional[str] = None
    watched_files: Optional[List[str]] = None
    payloads: Optional[List[dict]] = None
    iter_start: Optional[int] = None
    iter_end: Optional[int] = None
    ranks: Optional[List[int]] = None
    render_sig: Optional[Tuple[Tuple[str, float], ...]] = None
    timeline_cache: Optional[Dict[Tuple, bytes]] = None
    heatmap_cache: Optional[Dict[str, bytes]] = None
    timeline_maps: Optional[Dict[str, Tuple[Dict[int, Dict[Tuple[int, int], float]], Dict[int, Dict[Tuple[int, int], float]], int]]] = None
    timeline_history_keys: Optional[Dict[str, List[Tuple[int, int]]]] = None
    timeline_scale_bucket: Optional[int] = None
    timeline_scales: Optional[Dict[Tuple[str, str, int], Tuple[Optional[float], Optional[float]]]] = None


def _iter_json_files(inputs: Sequence[str]) -> List[str]:
    paths: List[str] = []
    for p in inputs:
        if os.path.isdir(p):
            per_rank = os.path.join(p, "per_rank_json")
            if os.path.isdir(per_rank):
                for name in os.listdir(per_rank):
                    if name.endswith(".json"):
                        paths.append(os.path.join(per_rank, name))
                continue
            for root, _dirs, files in os.walk(p):
                for name in files:
                    if re.match(r"rank_\d+\.json$", name):
                        paths.append(os.path.join(root, name))
        else:
            paths.append(p)
    uniq = sorted(set(paths))
    return [p for p in uniq if os.path.isfile(p)]


def _load_rank_payload(path: str) -> Optional[dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, dict) and isinstance(obj.get("records", None), list):
            return obj
    except Exception:
        return None
    return None


def _select_records(
    payload: dict,
    metric: str,
    iter_start: Optional[int],
    iter_end: Optional[int],
    mb_start: Optional[int],
    mb_end: Optional[int],
) -> Tuple[List[float], List[float]]:
    forward: List[float] = []
    backward: List[float] = []
    records = payload.get("records", []) or []
    for r in records:
        if not isinstance(r, dict):
            continue
        phase = r.get("phase", None)
        if phase not in ("forward_total", "backward_total"):
            continue
        it = r.get("iter", None)
        mb = r.get("mb", None)
        try:
            it_i = int(it)
            mb_i = int(mb)
        except Exception:
            continue
        if iter_start is not None and it_i < int(iter_start):
            continue
        if iter_end is not None and it_i >= int(iter_end):
            continue
        if mb_start is not None and mb_i < int(mb_start):
            continue
        if mb_end is not None and mb_i >= int(mb_end):
            continue
        v = r.get(metric, None)
        try:
            v_f = float(v)
        except Exception:
            continue
        if phase == "forward_total":
            forward.append(v_f)
        else:
            backward.append(v_f)
    return forward, backward


def _mean_or_nan(values: Sequence[float]) -> float:
    if not values:
        return float("nan")
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _infer_grid(n: int) -> Tuple[int, int]:
    if n <= 0:
        return 1, 1
    cols = int(math.ceil(math.sqrt(n)))
    rows = int(math.ceil(n / cols))
    return rows, cols


def _reshape_by_rank(values_by_rank: Dict[int, float], rows: int, cols: int) -> np.ndarray:
    mat = np.full((rows, cols), np.nan, dtype=np.float64)
    for rank, v in values_by_rank.items():
        idx = int(rank)
        r = idx // cols
        c = idx % cols
        if 0 <= r < rows and 0 <= c < cols:
            mat[r, c] = float(v)
    return mat


def _plot_single_heatmap(
    ax,
    mat: np.ndarray,
    title: str,
    unit: str,
    annotate: bool,
    cmap: str,
    vmin: Optional[float],
    vmax: Optional[float],
):
    im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest", aspect="equal")
    ax.set_title(title)
    ax.set_xlabel("rank % grid_cols")
    ax.set_ylabel("rank // grid_cols")
    ax.set_xticks([])
    ax.set_yticks([])
    if annotate:
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat[i, j]
                if np.isnan(v):
                    continue
                txt = f"{v:.1f}" if unit == "ms" else f"{v:.3f}"
                ax.text(j, i, txt, ha="center", va="center", fontsize=8, color="black")
    return im


def _to_unit(ms: float, unit: str) -> float:
    if unit == "ms":
        return float(ms)
    return float(ms) / 1000.0


def _max_iter_in_payloads(payloads: Sequence[dict]) -> Optional[int]:
    m: Optional[int] = None
    for pl in payloads:
        for r in pl.get("records", []) or []:
            if not isinstance(r, dict):
                continue
            it = r.get("iter", None)
            try:
                it_i = int(it)
            except Exception:
                continue
            if m is None or it_i > m:
                m = it_i
    return m


def _apply_tail_iters(iter_start: Optional[int], iter_end: Optional[int], payloads: Sequence[dict], tail_iters: Optional[int]) -> Tuple[Optional[int], Optional[int]]:
    if iter_start is not None or iter_end is not None:
        return iter_start, iter_end
    if tail_iters is None:
        return None, None
    max_it = _max_iter_in_payloads(payloads)
    if max_it is None:
        return None, None
    end = int(max_it) + 1
    start = max(0, end - int(tail_iters))
    return int(start), int(end)


def _build_rank_phase_maps(
    payloads: Sequence[dict],
    metric: str,
    iter_start: Optional[int],
    iter_end: Optional[int],
    mb_start: Optional[int],
    mb_end: Optional[int],
) -> Tuple[Dict[int, Dict[Tuple[int, int], float]], Dict[int, Dict[Tuple[int, int], float]], int]:
    forward_by_rank: Dict[int, Dict[Tuple[int, int], float]] = {}
    backward_by_rank: Dict[int, Dict[Tuple[int, int], float]] = {}
    world_size = 0

    for pl in payloads:
        try:
            rank = int(pl.get("rank", 0))
        except Exception:
            continue
        try:
            ws = int(pl.get("world_size", 0))
            world_size = max(world_size, ws)
        except Exception:
            pass

        fwd, bwd = _select_records(pl, metric, iter_start, iter_end, mb_start, mb_end)
        _ = (fwd, bwd)

        f_map = forward_by_rank.get(rank)
        if f_map is None:
            f_map = {}
            forward_by_rank[rank] = f_map
        b_map = backward_by_rank.get(rank)
        if b_map is None:
            b_map = {}
            backward_by_rank[rank] = b_map

        for r in pl.get("records", []) or []:
            if not isinstance(r, dict):
                continue
            phase = r.get("phase", None)
            if phase not in ("forward_total", "backward_total"):
                continue
            it = r.get("iter", None)
            mb = r.get("mb", None)
            try:
                it_i = int(it)
                mb_i = int(mb)
            except Exception:
                continue
            if iter_start is not None and it_i < int(iter_start):
                continue
            if iter_end is not None and it_i >= int(iter_end):
                continue
            if mb_start is not None and mb_i < int(mb_start):
                continue
            if mb_end is not None and mb_i >= int(mb_end):
                continue
            v = r.get(metric, None)
            try:
                v_f = float(v)
            except Exception:
                continue
            key = (it_i, mb_i)
            if phase == "forward_total":
                f_map[key] = v_f
            else:
                b_map[key] = v_f

    if world_size <= 0 and forward_by_rank:
        world_size = max(forward_by_rank.keys()) + 1
    if world_size <= 0 and backward_by_rank:
        world_size = max(backward_by_rank.keys()) + 1

    return forward_by_rank, backward_by_rank, int(world_size)


def _compute_timeline_segments(
    forward_by_rank: Dict[int, Dict[Tuple[int, int], float]],
    backward_by_rank: Dict[int, Dict[Tuple[int, int], float]],
    world_size: int,
    unit: str,
    sync_boundary: bool,
) -> Tuple[List[RankTimelineSegment], List[Tuple[int, int]], float]:
    ranks = sorted(set(list(forward_by_rank.keys()) + list(backward_by_rank.keys())))
    if not ranks:
        return [], [], 0.0

    if world_size <= 0:
        world_size = max(ranks) + 1

    keys: List[Tuple[int, int]] = []
    all_keys = set()
    for r in ranks:
        all_keys.update((forward_by_rank.get(r) or {}).keys())
        all_keys.update((backward_by_rank.get(r) or {}).keys())
    keys = sorted(all_keys, key=lambda k: (int(k[0]), int(k[1])))

    segments: List[RankTimelineSegment] = []
    t = 0.0
    if sync_boundary:
        for it, mb in keys:
            stage_f_durs: List[float] = []
            for r in ranks:
                v = (forward_by_rank.get(r) or {}).get((it, mb), 0.0)
                stage_f_durs.append(_to_unit(float(v), unit))
            for r, d in zip(ranks, stage_f_durs):
                segments.append(
                    RankTimelineSegment(rank=int(r), start=float(t), duration=float(d), phase="forward", iter=int(it), mb=int(mb))
                )
            t += float(max(stage_f_durs) if stage_f_durs else 0.0)

            stage_b_durs: List[float] = []
            for r in ranks:
                v = (backward_by_rank.get(r) or {}).get((it, mb), 0.0)
                stage_b_durs.append(_to_unit(float(v), unit))
            for r, d in zip(ranks, stage_b_durs):
                segments.append(
                    RankTimelineSegment(rank=int(r), start=float(t), duration=float(d), phase="backward", iter=int(it), mb=int(mb))
                )
            t += float(max(stage_b_durs) if stage_b_durs else 0.0)
    else:
        per_rank_t: Dict[int, float] = {int(r): 0.0 for r in ranks}
        for it, mb in keys:
            for r in ranks:
                d = _to_unit(float((forward_by_rank.get(r) or {}).get((it, mb), 0.0)), unit)
                segments.append(
                    RankTimelineSegment(rank=int(r), start=float(per_rank_t[int(r)]), duration=float(d), phase="forward", iter=int(it), mb=int(mb))
                )
                per_rank_t[int(r)] += float(d)
                d2 = _to_unit(float((backward_by_rank.get(r) or {}).get((it, mb), 0.0)), unit)
                segments.append(
                    RankTimelineSegment(rank=int(r), start=float(per_rank_t[int(r)]), duration=float(d2), phase="backward", iter=int(it), mb=int(mb))
                )
                per_rank_t[int(r)] += float(d2)
        t = float(max(per_rank_t.values()) if per_rank_t else 0.0)

    return segments, keys, float(t)


def _render_rank_timeline_png_bytes(
    segments: Sequence[RankTimelineSegment],
    keys: Sequence[Tuple[int, int]],
    total_time: float,
    unit: str,
    annotate: bool,
    show_mb_separators: bool,
    sync_boundary: bool,
) -> bytes:
    if not segments:
        raise SystemExit("timeline 没有可绘制的 segment")

    ranks = sorted(set(int(s.rank) for s in segments))
    rank_to_y = {r: i for i, r in enumerate(ranks)}

    height = 0.8
    fig_h = max(3.0, 0.5 + len(ranks) * 0.35)
    fig_w = max(10.0, 6.0 + min(24.0, total_time / (20.0 if unit == "ms" else 0.02)))
    fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h), constrained_layout=True)

    colors = {"forward": "#7f7f7f", "backward": "#ff7f0e"}
    for s in segments:
        if s.duration <= 0:
            continue
        y = rank_to_y[int(s.rank)]
        ax.broken_barh([(float(s.start), float(s.duration))], (y - height / 2.0, height), facecolors=colors.get(s.phase, "#1f77b4"))

    ax.set_yticks([rank_to_y[r] for r in ranks])
    ax.set_yticklabels([f"rank[{r}]" for r in ranks])
    ax.set_xlabel(f"time ({unit})")
    ax.set_xlim(0.0, max(total_time, 1e-9))
    ax.set_title("Microbatch timeline by rank (forward/backward)")

    handles = [
        matplotlib.patches.Patch(color=colors["forward"], label="forward"),
        matplotlib.patches.Patch(color=colors["backward"], label="backward"),
    ]
    ax.legend(handles=handles, loc="upper right")

    if show_mb_separators and sync_boundary:
        t = 0.0
        for it, mb in keys:
            f_durs = [seg.duration for seg in segments if seg.iter == it and seg.mb == mb and seg.phase == "forward"]
            b_durs = [seg.duration for seg in segments if seg.iter == it and seg.mb == mb and seg.phase == "backward"]
            t += float(max(f_durs) if f_durs else 0.0)
            ax.axvline(t, color="#cccccc", linewidth=0.6)
            t += float(max(b_durs) if b_durs else 0.0)
            ax.axvline(t, color="#dddddd", linewidth=0.4)

    if annotate and sync_boundary:
        t = 0.0
        for it, mb in keys:
            f_durs = [seg.duration for seg in segments if seg.iter == it and seg.mb == mb and seg.phase == "forward"]
            b_durs = [seg.duration for seg in segments if seg.iter == it and seg.mb == mb and seg.phase == "backward"]
            stage_f = float(max(f_durs) if f_durs else 0.0)
            stage_b = float(max(b_durs) if b_durs else 0.0)
            mid = t + stage_f + stage_b / 2.0
            ax.text(mid, len(ranks) - 0.2, f"it{it}/mb{mb}", ha="center", va="bottom", fontsize=7, rotation=0)
            t += stage_f + stage_b

    buf = io.BytesIO()
    fig.savefig(buf, dpi=200, format="png")
    plt.close(fig)
    return buf.getvalue()


def _collect_keys(
    forward_by_rank: Dict[int, Dict[Tuple[int, int], float]],
    backward_by_rank: Dict[int, Dict[Tuple[int, int], float]],
) -> List[Tuple[int, int]]:
    all_keys = set()
    for m in (forward_by_rank, backward_by_rank):
        for _rank, per_key in (m or {}).items():
            all_keys.update((per_key or {}).keys())
    return sorted(all_keys, key=lambda k: (int(k[0]), int(k[1])))


def _robust_vmin_vmax(mat: np.ndarray, robust: bool) -> Tuple[Optional[float], Optional[float]]:
    finite = mat[np.isfinite(mat)]
    if finite.size == 0:
        return None, None
    if robust and finite.size >= 8:
        lo = float(np.percentile(finite, 5.0))
        hi = float(np.percentile(finite, 95.0))
        if math.isfinite(lo) and math.isfinite(hi) and hi > lo:
            return lo, hi
    lo2 = float(np.min(finite))
    hi2 = float(np.max(finite))
    if math.isfinite(lo2) and math.isfinite(hi2) and hi2 > lo2:
        return lo2, hi2
    return None, None


def _build_rank_mb_matrix(
    forward_by_rank: Dict[int, Dict[Tuple[int, int], float]],
    backward_by_rank: Dict[int, Dict[Tuple[int, int], float]],
    ranks: Sequence[int],
    keys: Sequence[Tuple[int, int]],
    unit: str,
    phase: str,
) -> Tuple[List[int], np.ndarray]:
    ranks2 = [int(r) for r in ranks]
    if not ranks2:
        ranks2 = sorted(set(list(forward_by_rank.keys()) + list(backward_by_rank.keys())))
    if not ranks2:
        raise SystemExit("timeline 没有可绘制的 rank")
    if not keys:
        raise SystemExit("timeline 没有可绘制的 microbatch keys")

    r_n = len(ranks2)
    k_n = len(keys)
    mat = np.full((r_n, k_n), np.nan, dtype=np.float64)

    src = forward_by_rank if phase == "forward" else backward_by_rank
    for i, r in enumerate(ranks2):
        per_key = (src or {}).get(int(r), None) or {}
        for j, k in enumerate(keys):
            v = per_key.get(k, None)
            if v is None:
                continue
            mat[i, j] = _to_unit(float(v), unit)
    return ranks2, mat


def _render_rank_mb_phase_png_bytes(
    forward_by_rank: Dict[int, Dict[Tuple[int, int], float]],
    backward_by_rank: Dict[int, Dict[Tuple[int, int], float]],
    ranks: Sequence[int],
    keys: Sequence[Tuple[int, int]],
    unit: str,
    annotate: bool,
    cmap: str,
    phase: str,
    robust_scale: bool,
    scale_vmin: Optional[float],
    scale_vmax: Optional[float],
) -> bytes:
    ranks2, mat = _build_rank_mb_matrix(forward_by_rank, backward_by_rank, ranks, keys, unit=unit, phase=phase)
    r_n, k_n = mat.shape

    max_cols = 80.0
    fig_w = max(10.0, min(max_cols, 2.0 + 0.28 * k_n))
    fig_h = max(4.0, 1.5 + 0.22 * r_n)
    fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h), constrained_layout=True)

    if scale_vmin is not None and scale_vmax is not None:
        vmin, vmax = float(scale_vmin), float(scale_vmax)
    else:
        vmin, vmax = _robust_vmin_vmax(mat, robust=bool(robust_scale))
    im = ax.imshow(mat, aspect="auto", interpolation="nearest", cmap=cmap, vmin=vmin, vmax=vmax, origin="lower")
    ax.set_title(f"{phase.capitalize()} ({unit})")
    ax.set_ylabel("rank")
    ax.set_xlabel("microbatch (iter/mb)")
    ax.set_yticks(list(range(r_n)))
    ax.set_yticklabels([f"rank[{r}]" for r in ranks2])

    step = max(1, k_n // 24)
    xticks = list(range(0, k_n, step))
    xlabels = [f"it{keys[j][0]}/mb{keys[j][1]}" for j in xticks]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, rotation=45, ha="right", fontsize=8)

    for j in range(1, k_n):
        ax.axvline(j - 0.5, color="#e6e6e6", linewidth=0.8, alpha=0.9)

    last_it = keys[0][0]
    for j, (it, _mb) in enumerate(keys):
        if it != last_it:
            ax.axvline(j - 0.5, color="#b5b5b5", linewidth=1.4, alpha=0.95)
            last_it = it

    for i in range(1, r_n):
        ax.axhline(i - 0.5, color="#4a4a4a", linewidth=1.2, alpha=0.9)

    if annotate and (r_n * k_n) <= 1200:
        fmt = "{:.1f}" if unit == "ms" else "{:.3f}"
        for i in range(r_n):
            for j in range(k_n):
                v = mat[i, j]
                if np.isfinite(v):
                    ax.text(j, i, fmt.format(float(v)), ha="center", va="center", fontsize=6, color="black")

    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    fig.suptitle("Rank × microbatch heatmap")

    buf = io.BytesIO()
    fig.savefig(buf, dpi=200, format="png")
    plt.close(fig)
    return buf.getvalue()


def _render_heatmap_png_bytes(
    avgs: Sequence[RankAverages],
    world_size: int,
    unit: str,
    cmap: str,
    annotate: bool,
    grid_rows: Optional[int],
    grid_cols: Optional[int],
    shared_scale: bool,
    metric: str,
) -> bytes:
    if not avgs:
        raise SystemExit("heatmap 没有可绘制的数据")

    n = max(int(world_size), max(int(a.rank) for a in avgs) + 1)
    rows = int(grid_rows) if grid_rows is not None else None
    cols = int(grid_cols) if grid_cols is not None else None
    if rows is None or cols is None:
        rows2, cols2 = _infer_grid(n)
        rows = rows or rows2
        cols = cols or cols2

    fwd_by_rank: Dict[int, float] = {int(a.rank): _to_unit(float(a.forward_ms), unit) for a in avgs}
    bwd_by_rank: Dict[int, float] = {int(a.rank): _to_unit(float(a.backward_ms), unit) for a in avgs}
    fwd_mat = _reshape_by_rank(fwd_by_rank, int(rows), int(cols))
    bwd_mat = _reshape_by_rank(bwd_by_rank, int(rows), int(cols))

    if shared_scale:
        comb = np.concatenate([fwd_mat.reshape(-1), bwd_mat.reshape(-1)], axis=0)
        finite = comb[np.isfinite(comb)]
        vmin = float(np.min(finite)) if finite.size else None
        vmax = float(np.max(finite)) if finite.size else None
        vmin_f, vmax_f, vmin_b, vmax_b = vmin, vmax, vmin, vmax
    else:
        vmin_f = float(np.nanmin(fwd_mat)) if np.isfinite(fwd_mat).any() else None
        vmax_f = float(np.nanmax(fwd_mat)) if np.isfinite(fwd_mat).any() else None
        vmin_b = float(np.nanmin(bwd_mat)) if np.isfinite(bwd_mat).any() else None
        vmax_b = float(np.nanmax(bwd_mat)) if np.isfinite(bwd_mat).any() else None

    fig, axes = plt.subplots(1, 2, figsize=(max(6, int(cols) * 1.2), max(3, int(rows) * 1.2)), constrained_layout=True)
    im0 = _plot_single_heatmap(
        axes[0],
        fwd_mat,
        f"Forward avg ({metric}, {unit})",
        unit,
        annotate,
        cmap,
        vmin_f,
        vmax_f,
    )
    im1 = _plot_single_heatmap(
        axes[1],
        bwd_mat,
        f"Backward avg ({metric}, {unit})",
        unit,
        annotate,
        cmap,
        vmin_b,
        vmax_b,
    )
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    buf = io.BytesIO()
    fig.savefig(buf, dpi=200, format="png")
    plt.close(fig)
    return buf.getvalue()


def _file_mtime_signature(files: Sequence[str]) -> Tuple[Tuple[str, float], ...]:
    sig: List[Tuple[str, float]] = []
    for p in files:
        try:
            sig.append((p, float(os.path.getmtime(p))))
        except Exception:
            sig.append((p, -1.0))
    return tuple(sig)


def _load_payloads_from_inputs(inputs: Sequence[str]) -> Tuple[List[dict], List[str]]:
    files = _iter_json_files(inputs)
    payloads: List[dict] = []
    for p in files:
        pl = _load_rank_payload(p)
        if pl is not None:
            payloads.append(pl)
    return payloads, files


def _available_ranks(payloads: Sequence[dict]) -> List[int]:
    ranks: List[int] = []
    for pl in payloads:
        try:
            ranks.append(int(pl.get("rank", 0)))
        except Exception:
            continue
    return sorted(set(ranks))


def _filter_rank_phase_maps_by_set(
    forward_by_rank: Dict[int, Dict[Tuple[int, int], float]],
    backward_by_rank: Dict[int, Dict[Tuple[int, int], float]],
    keep_ranks: Sequence[int],
) -> Tuple[Dict[int, Dict[Tuple[int, int], float]], Dict[int, Dict[Tuple[int, int], float]]]:
    keep = set(int(r) for r in keep_ranks)
    out_f: Dict[int, Dict[Tuple[int, int], float]] = {}
    out_b: Dict[int, Dict[Tuple[int, int], float]] = {}
    for r in keep:
        if r in forward_by_rank:
            out_f[r] = forward_by_rank[r]
        if r in backward_by_rank:
            out_b[r] = backward_by_rank[r]
    return out_f, out_b


def _serve_live_dashboard(args) -> int:
    state = LiveRenderState()
    lock = threading.Lock()
    stop_evt = threading.Event()
    last_sig: Optional[Tuple[Tuple[str, float], ...]] = None

    def _rebuild():
        nonlocal last_sig
        payloads, files = _load_payloads_from_inputs(args.input)
        sig = _file_mtime_signature(files)
        if last_sig is not None and sig == last_sig and state.payloads is not None:
            return
        last_sig = sig

        iter_start, iter_end = _apply_tail_iters(args.iter_start, args.iter_end, payloads, args.tail_iters)

        timeline_maps: Dict[str, Tuple[Dict[int, Dict[Tuple[int, int], float]], Dict[int, Dict[Tuple[int, int], float]], int]] = {}
        timeline_history_keys: Dict[str, List[Tuple[int, int]]] = {}
        timeline_scale_bucket = 100
        timeline_scales: Dict[Tuple[str, str, int], Tuple[Optional[float], Optional[float]]] = {}
        hist_limit = int(getattr(args, "timeline_history", 100))
        if hist_limit <= 0:
            hist_limit = 100
        for metric in ("gpu_ms", "cpu_ms"):
            f_map, b_map, ws = _build_rank_phase_maps(payloads, metric, iter_start, iter_end, args.mb_start, args.mb_end)
            timeline_maps[metric] = (f_map, b_map, ws)
            keys_all = _collect_keys(f_map, b_map)
            if len(keys_all) > hist_limit:
                keys_all = keys_all[-hist_limit:]
            timeline_history_keys[metric] = keys_all
            if not keys_all:
                continue
            key_to_idx: Dict[Tuple[int, int], int] = {k: i for i, k in enumerate(keys_all)}
            bucket_values: Dict[Tuple[str, str, int], List[float]] = {}
            for phase in ("forward", "backward"):
                src = f_map if phase == "forward" else b_map
                for per_key in (src or {}).values():
                    for k, v in (per_key or {}).items():
                        idx = key_to_idx.get(k, None)
                        if idx is None:
                            continue
                        bucket_id = int(idx) // int(timeline_scale_bucket)
                        try:
                            bucket_values.setdefault((metric, phase, bucket_id), []).append(_to_unit(float(v), args.unit))
                        except Exception:
                            continue
            bucket_count = int((len(keys_all) + int(timeline_scale_bucket) - 1) // int(timeline_scale_bucket))
            for phase in ("forward", "backward"):
                for bucket_id in range(bucket_count):
                    values = bucket_values.get((metric, phase, int(bucket_id)), []) or []
                    if not values:
                        timeline_scales[(metric, phase, int(bucket_id))] = (None, None)
                        continue
                    arr = np.asarray(values, dtype=np.float64)
                    finite = arr[np.isfinite(arr)]
                    if finite.size == 0:
                        timeline_scales[(metric, phase, int(bucket_id))] = (None, None)
                        continue
                    if bool(getattr(args, "timeline_robust_scale", True)) and finite.size >= 8:
                        lo = float(np.percentile(finite, 5.0))
                        hi = float(np.percentile(finite, 95.0))
                    else:
                        lo = float(np.min(finite))
                        hi = float(np.max(finite))
                    if not (math.isfinite(lo) and math.isfinite(hi) and hi > lo):
                        timeline_scales[(metric, phase, int(bucket_id))] = (None, None)
                        continue
                    if lo < 0.0:
                        lo = 0.0
                    timeline_scales[(metric, phase, int(bucket_id))] = (float(lo), float(hi))

        with lock:
            state.payloads = payloads
            state.last_update_s = float(time.time())
            state.last_error = None
            state.watched_files = files
            state.iter_start = iter_start
            state.iter_end = iter_end
            state.ranks = _available_ranks(payloads)
            state.render_sig = sig
            state.timeline_cache = {}
            state.heatmap_cache = {}
            state.timeline_maps = timeline_maps
            state.timeline_history_keys = timeline_history_keys
            state.timeline_scale_bucket = int(timeline_scale_bucket)
            state.timeline_scales = timeline_scales

    def _updater():
        while not stop_evt.is_set():
            try:
                _rebuild()
            except Exception as e:
                with lock:
                    state.last_error = str(e)
                    state.last_update_s = float(time.time())
            stop_evt.wait(float(max(0.2, args.refresh_sec)))

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            parsed = urllib.parse.urlparse(self.path)
            path = parsed.path
            if path == "/":
                self._send_index()
                return
            if path == "/timeline":
                self._send_timeline_page(parsed.query)
                return
            if path == "/heatmap":
                self._send_heatmap_page(parsed.query)
                return
            if path == "/timeline.png":
                self._send_timeline_png(parsed.query)
                return
            if path == "/timeline_meta.json":
                self._send_timeline_meta(parsed.query)
                return
            if path == "/heatmap.png":
                self._send_heatmap_png(parsed.query)
                return
            if path == "/status.json":
                self._send_status()
                return
            self.send_response(404)
            self.end_headers()

        def log_message(self, format, *args):
            return

        def _send_index(self):
            with lock:
                last = state.last_update_s
                err = state.last_error
                ranks = state.ranks or []
            refresh_ms = int(max(200, float(args.refresh_sec) * 1000.0))
            html = f"""
<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <title>Module Hook Profiler Live</title>
    <style>
      body {{ font-family: sans-serif; margin: 16px; }}
      .row {{ display: flex; flex-direction: column; gap: 16px; }}
      img {{ max-width: 100%; height: auto; border: 1px solid #ddd; }}
      .meta {{ color: #444; font-size: 13px; }}
      .err {{ color: #b00020; }}
      a {{ color: #0b57d0; text-decoration: none; }}
      a:hover {{ text-decoration: underline; }}
      .nav {{ display: flex; gap: 12px; margin: 12px 0; }}
    </style>
  </head>
  <body>
    <div class="meta">
      <div>refresh: {args.refresh_sec:.2f}s | metric: {args.metric} | unit: {args.unit} | tail_iters: {args.tail_iters}</div>
      <div>last_update_s: {last:.3f} {('<span class="err">error: ' + err + '</span>') if err else ''}</div>
      <div>ranks: {len(ranks)} ({(str(ranks[:8]) + (' ...' if len(ranks) > 8 else ''))})</div>
    </div>
    <div class="nav">
      <a href="/timeline">Timeline</a>
      <a href="/heatmap">Heatmap</a>
    </div>
    <script>
      function tick() {{
        const t = Date.now();
        fetch("/status.json?t=" + t).then(r => r.json()).then(js => {{
          const meta = document.querySelector(".meta");
          if (meta && js && js.last_update_s !== undefined) {{
            meta.children[1].innerHTML = "last_update_s: " + js.last_update_s.toFixed(3) + (js.last_error ? (' <span class=\\"err\\">error: ' + js.last_error + '</span>') : "");
          }}
        }}).catch(_ => {{}});
      }}
      setInterval(tick, {refresh_ms});
      tick();
    </script>
  </body>
</html>
"""
            body = html.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _parse_metric(self, qs: dict) -> str:
            metric = str(qs.get("metric", [str(args.metric)])[0] or str(args.metric))
            if metric not in ("gpu_ms", "cpu_ms"):
                metric = str(args.metric)
            return metric

        def _parse_phase(self, qs: dict) -> str:
            phase = str(qs.get("phase", [str(getattr(args, "timeline_phase", "forward"))])[0] or "forward")
            if phase not in ("forward", "backward"):
                phase = "forward"
            return phase

        def _send_timeline_page(self, query: str):
            qs = urllib.parse.parse_qs(query or "")
            page = int(qs.get("page", ["0"])[0])
            page_size = int(qs.get("page_size", [str(args.page_size)])[0])
            phase = self._parse_phase(qs)
            metric = self._parse_metric(qs)
            window = int(qs.get("window", [str(getattr(args, "timeline_window", 10))])[0])
            window = max(1, min(200, int(window)))
            with lock:
                ranks = state.ranks or []
                hist_keys = (state.timeline_history_keys or {}).get(metric, []) or []
            max_page = max(0, (len(ranks) - 1) // max(1, page_size)) if ranks else 0
            page = max(0, min(page, max_page))
            rank_start = page * page_size
            rank_end = min(len(ranks), (page + 1) * page_size)
            shown = ranks[rank_start:rank_end] if ranks else []
            key_count = int(len(hist_keys))
            max_start = max(0, int(key_count) - int(window))
            refresh_ms = int(max(200, float(args.refresh_sec) * 1000.0))
            html = f"""
<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <title>Timeline</title>
    <style>
      body {{ font-family: sans-serif; margin: 16px; }}
      img {{ max-width: 100%; height: auto; border: 1px solid #ddd; }}
      .meta {{ color: #444; font-size: 13px; margin-bottom: 10px; }}
      .nav {{ display: flex; gap: 12px; margin: 12px 0; align-items: center; }}
      .btn {{ padding: 6px 10px; border: 1px solid #ccc; border-radius: 6px; cursor: pointer; background: #fff; }}
      .btn:disabled {{ opacity: 0.4; cursor: default; }}
      .pill {{ padding: 6px 10px; border: 1px solid #ccc; border-radius: 999px; }}
      .slider {{ width: min(900px, 100%); }}
      a {{ color: #0b57d0; text-decoration: none; }}
      a:hover {{ text-decoration: underline; }}
    </style>
  </head>
  <body>
    <div class="nav">
      <a href="/">Home</a>
      <a href="/heatmap?metric={metric}">Heatmap</a>
      <a href="/timeline?metric=gpu_ms&phase={phase}&page={page}&page_size={page_size}&window={window}">GPU</a>
      <a href="/timeline?metric=cpu_ms&phase={phase}&page={page}&page_size={page_size}&window={window}">CPU</a>
      <a href="/timeline?metric={metric}&phase=forward&page={page}&page_size={page_size}&window={window}">Forward</a>
      <a href="/timeline?metric={metric}&phase=backward&page={page}&page_size={page_size}&window={window}">Backward</a>
      <span class="meta">metric {metric} | phase {phase} | page {page}/{max_page} | page_size {page_size} | window {window} | key_count {key_count} | ranks {shown[:4]}{" ..." if len(shown) > 4 else ""} | unit {args.unit}</span>
    </div>
    <div class="nav">
      <button class="btn" id="prev" onclick="go(-1)">Prev</button>
      <button class="btn" id="next" onclick="go(1)">Next</button>
    </div>
    <div class="nav">
      <span class="pill">microbatch window: last {window}</span>
      <input class="slider" id="mbStart" type="range" min="0" max="{max_start}" value="{max_start}" step="1" />
      <span class="meta" id="mbLabel">start {max_start} / {max_start}</span>
    </div>
    <img id="img" src="/timeline.png?page={page}&page_size={page_size}&phase={phase}&metric={metric}&start={max_start}&window={window}" />
    <script>
      const page = {page};
      const maxPage = {max_page};
      const pageSize = {page_size};
      const phase = "{phase}";
      const metric = "{metric}";
      const windowSize = {window};
      const slider = document.getElementById("mbStart");
      const label = document.getElementById("mbLabel");
      let followTail = true;
      document.getElementById("prev").disabled = (page <= 0);
      document.getElementById("next").disabled = (page >= maxPage);
      function go(delta) {{
        const p = Math.max(0, Math.min(maxPage, page + delta));
        window.location = "/timeline?metric=" + metric + "&phase=" + phase + "&page=" + p + "&page_size=" + pageSize + "&window=" + windowSize + "&start=" + slider.value;
      }}
      function setLabel(maxStart) {{
        label.textContent = "start " + slider.value + " / " + maxStart;
      }}
      function refreshMetaAndMaybeFollowTail() {{
        const t = Date.now();
        return fetch("/timeline_meta.json?metric=" + metric + "&page=" + page + "&page_size=" + pageSize + "&window=" + windowSize + "&t=" + t)
          .then(r => r.json())
          .then(js => {{
            const maxStart = (js && js.max_start !== undefined) ? js.max_start : 0;
            slider.max = String(maxStart);
            if (followTail) {{
              slider.value = String(maxStart);
            }} else {{
              if (Number(slider.value) > maxStart) {{
                slider.value = String(maxStart);
              }}
            }}
            setLabel(maxStart);
          }})
          .catch(_ => {{
            setLabel(Number(slider.max || 0));
          }});
      }}
      function render() {{
        const t = Date.now();
        const start = slider.value;
        document.getElementById("img").src = "/timeline.png?page=" + page + "&page_size=" + pageSize + "&phase=" + phase + "&metric=" + metric + "&start=" + start + "&window=" + windowSize + "&t=" + t;
      }}
      slider.addEventListener("input", () => {{
        followTail = false;
        setLabel(Number(slider.max || 0));
        render();
      }});
      function tick() {{
        refreshMetaAndMaybeFollowTail().then(render);
      }}
      setInterval(tick, {refresh_ms});
      tick();
    </script>
  </body>
</html>
"""
            body = html.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_heatmap_page(self, query: str):
            qs = urllib.parse.parse_qs(query or "")
            metric = self._parse_metric(qs)
            refresh_ms = int(max(200, float(args.refresh_sec) * 1000.0))
            html = f"""
<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <title>Heatmap</title>
    <style>
      body {{ font-family: sans-serif; margin: 16px; }}
      img {{ max-width: 100%; height: auto; border: 1px solid #ddd; }}
      .meta {{ color: #444; font-size: 13px; margin-bottom: 10px; }}
      .nav {{ display: flex; gap: 12px; margin: 12px 0; align-items: center; }}
      a {{ color: #0b57d0; text-decoration: none; }}
      a:hover {{ text-decoration: underline; }}
    </style>
  </head>
  <body>
    <div class="nav">
      <a href="/">Home</a>
      <a href="/timeline?metric={metric}">Timeline</a>
      <a href="/heatmap?metric=gpu_ms">GPU</a>
      <a href="/heatmap?metric=cpu_ms">CPU</a>
      <span class="meta">metric {metric} | unit {args.unit}</span>
    </div>
    <img id="img" src="/heatmap.png?metric={metric}" />
    <script>
      const metric = "{metric}";
      function tick() {{
        const t = Date.now();
        document.getElementById("img").src = "/heatmap.png?metric=" + metric + "&t=" + t;
      }}
      setInterval(tick, {refresh_ms});
      tick();
    </script>
  </body>
</html>
"""
            body = html.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_timeline_meta(self, query: str):
            qs = urllib.parse.parse_qs(query or "")
            page = int(qs.get("page", ["0"])[0])
            page_size = int(qs.get("page_size", [str(args.page_size)])[0])
            metric = self._parse_metric(qs)
            window = int(qs.get("window", [str(getattr(args, "timeline_window", 10))])[0])
            window = max(1, min(200, int(window)))
            with lock:
                ranks = list(state.ranks or [])
                hist_keys = (state.timeline_history_keys or {}).get(metric, []) or []
            _ = (page, page_size, ranks)
            key_count = int(len(hist_keys))
            max_start = max(0, int(key_count) - int(window))
            payload = {"key_count": int(key_count), "max_start": int(max_start)}
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store, max-age=0")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                return

        def _send_timeline_png(self, query: str):
            qs = urllib.parse.parse_qs(query or "")
            page = int(qs.get("page", ["0"])[0])
            page_size = int(qs.get("page_size", [str(args.page_size)])[0])
            phase = self._parse_phase(qs)
            metric = self._parse_metric(qs)
            window = int(qs.get("window", [str(getattr(args, "timeline_window", 10))])[0])
            window = max(1, min(200, int(window)))
            start_raw = qs.get("start", [None])[0]
            with lock:
                payloads = list(state.payloads or [])
                iter_start = state.iter_start
                iter_end = state.iter_end
                ranks = list(state.ranks or [])
                sig = state.render_sig
                cache = state.timeline_cache or {}
                maps = (state.timeline_maps or {}).get(metric, None)
                hist_keys = (state.timeline_history_keys or {}).get(metric, []) or []
                scale_bucket = int(state.timeline_scale_bucket or 100)
                scales = state.timeline_scales or {}
            if not payloads:
                self.send_response(503)
                self.end_headers()
                return
            if maps is None or not hist_keys:
                self.send_response(503)
                self.end_headers()
                return
            max_page = max(0, (len(ranks) - 1) // max(1, page_size)) if ranks else 0
            page = max(0, min(page, max_page))
            start_q = None
            try:
                if start_raw is not None:
                    start_q = int(start_raw)
            except Exception:
                start_q = None
            cache_key = (int(page), int(page_size), str(phase), str(metric), int(start_q) if start_q is not None else -1, int(window))
            cached = cache.get(cache_key, None)
            if cached is not None:
                data = cached
            else:
                rank_start = page * page_size
                rank_end = min(len(ranks), (page + 1) * page_size)
                page_ranks = ranks[rank_start:rank_end] if ranks else []

                f_map, b_map, _ws = maps
                _ = (iter_start, iter_end, payloads)
                max_start = max(0, len(hist_keys) - int(window))
                if start_q is None:
                    start = max_start
                else:
                    start = max(0, min(int(start_q), int(max_start)))
                keys = hist_keys[int(start) : int(start) + int(window)]
                bucket_id = int(int(start) // max(1, int(scale_bucket)))
                scale_vmin, scale_vmax = scales.get((str(metric), str(phase), int(bucket_id)), (None, None))
                data = _render_rank_mb_phase_png_bytes(
                    forward_by_rank=f_map,
                    backward_by_rank=b_map,
                    ranks=page_ranks,
                    keys=keys,
                    unit=args.unit,
                    annotate=bool(args.annotate),
                    cmap=args.cmap,
                    phase=str(phase),
                    robust_scale=bool(getattr(args, "timeline_robust_scale", True)),
                    scale_vmin=scale_vmin,
                    scale_vmax=scale_vmax,
                )
                with lock:
                    if state.render_sig == sig and state.timeline_cache is not None:
                        state.timeline_cache[cache_key] = data
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Cache-Control", "no-store, max-age=0")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            try:
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError):
                return

        def _send_heatmap_png(self, query: str):
            qs = urllib.parse.parse_qs(query or "")
            metric = self._parse_metric(qs)
            with lock:
                payloads = list(state.payloads or [])
                iter_start = state.iter_start
                iter_end = state.iter_end
                sig = state.render_sig
                cache = state.heatmap_cache or {}
            if not payloads:
                self.send_response(503)
                self.end_headers()
                return
            cached = cache.get(metric, None)
            if cached is not None:
                data = cached
            else:
                avgs, world_size = build_rank_averages(args.input, metric, iter_start, iter_end, args.mb_start, args.mb_end)
                data = _render_heatmap_png_bytes(
                    avgs=avgs,
                    world_size=world_size,
                    unit=args.unit,
                    cmap=args.cmap,
                    annotate=bool(args.annotate),
                    grid_rows=args.grid_rows,
                    grid_cols=args.grid_cols,
                    shared_scale=bool(args.shared_scale),
                    metric=metric,
                )
                with lock:
                    if state.render_sig == sig:
                        if state.heatmap_cache is not None:
                            state.heatmap_cache[metric] = data
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Cache-Control", "no-store, max-age=0")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            try:
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError):
                return

        def _send_status(self):
            with lock:
                payload = {
                    "last_update_s": float(state.last_update_s),
                    "last_error": state.last_error,
                    "watched_files": state.watched_files or [],
                    "iter_start": state.iter_start,
                    "iter_end": state.iter_end,
                    "rank_count": len(state.ranks or []),
                }
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store, max-age=0")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    t = threading.Thread(target=_updater, daemon=True)
    t.start()

    try:
        _rebuild()
    except Exception as e:
        with lock:
            state.last_error = str(e)
            state.last_update_s = float(time.time())

    server = ThreadingHTTPServer((args.host, int(args.port)), Handler)
    try:
        server.serve_forever()
    finally:
        stop_evt.set()
        try:
            server.server_close()
        except Exception:
            pass
    return 0


def build_rank_averages(
    inputs: Sequence[str],
    metric: str,
    iter_start: Optional[int],
    iter_end: Optional[int],
    mb_start: Optional[int],
    mb_end: Optional[int],
) -> Tuple[List[RankAverages], int]:
    json_files = _iter_json_files(inputs)
    payloads: List[dict] = []
    for p in json_files:
        pl = _load_rank_payload(p)
        if pl is not None:
            payloads.append(pl)

    ranks: Dict[int, RankAverages] = {}
    world_size = 0
    for pl in payloads:
        try:
            rank = int(pl.get("rank", 0))
        except Exception:
            continue
        try:
            ws = int(pl.get("world_size", 0))
            world_size = max(world_size, ws)
        except Exception:
            pass
        fwd, bwd = _select_records(pl, metric, iter_start, iter_end, mb_start, mb_end)
        ranks[rank] = RankAverages(rank=rank, forward_ms=_mean_or_nan(fwd), backward_ms=_mean_or_nan(bwd))

    if world_size <= 0 and ranks:
        world_size = max(ranks.keys()) + 1

    out = [ranks[k] for k in sorted(ranks.keys())]
    return out, int(world_size)


def _demo_payloads(world_size: int = 2, iters: int = 2, mbs: int = 4) -> List[dict]:
    payloads = []
    rng = np.random.default_rng(0)
    for rank in range(world_size):
        recs = []
        base_f = 35.0 + rank * 10.0
        base_b = 60.0 + rank * 12.0
        for it in range(iters):
            for mb in range(1, mbs + 1):
                recs.append({"phase": "forward_total", "iter": it, "mb": mb, "gpu_ms": float(base_f + rng.normal(0, 2.0))})
                recs.append({"phase": "backward_total", "iter": it, "mb": mb, "gpu_ms": float(base_b + rng.normal(0, 3.0))})
        payloads.append({"rank": rank, "world_size": world_size, "records": recs})
    return payloads


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", nargs="*", default=["."], help="rank_*.json 文件或包含 per_rank_json 的目录")
    ap.add_argument("--output", default="module_hook_profile_heatmap.png", help="输出图片路径")
    ap.add_argument("--mode", default="heatmap", choices=["heatmap", "timeline", "both", "serve"])
    ap.add_argument("--metric", default="gpu_ms", choices=["gpu_ms", "cpu_ms"], help="使用哪个字段聚合")
    ap.add_argument("--unit", default="s", choices=["ms", "s"], help="显示单位")
    ap.add_argument("--timeline-phase", default="forward", choices=["forward", "backward"], help="timeline 渲染 forward 或 backward")
    ap.add_argument("--timeline-robust-scale", action="store_true")
    ap.add_argument("--no-timeline-robust-scale", dest="timeline_robust_scale", action="store_false")
    ap.set_defaults(timeline_robust_scale=True)
    ap.add_argument("--timeline-window", type=int, default=10)
    ap.add_argument("--timeline-history", type=int, default=100)
    ap.add_argument("--iter-start", type=int, default=None)
    ap.add_argument("--iter-end", type=int, default=None)
    ap.add_argument("--mb-start", type=int, default=None)
    ap.add_argument("--mb-end", type=int, default=None)
    ap.add_argument("--tail-iters", type=int, default=None)
    ap.add_argument("--grid-rows", type=int, default=None)
    ap.add_argument("--grid-cols", type=int, default=None)
    ap.add_argument("--annotate", action="store_true")
    ap.add_argument("--no-annotate", dest="annotate", action="store_false")
    ap.set_defaults(annotate=True)
    ap.add_argument("--cmap", default="Reds")
    ap.add_argument("--shared-scale", action="store_true")
    ap.add_argument("--sync-boundary", action="store_true", help="对齐每个微批次阶段起点（用 max 作为边界）")
    ap.add_argument("--no-sync-boundary", dest="sync_boundary", action="store_false")
    ap.set_defaults(sync_boundary=True)
    ap.add_argument("--mb-separators", action="store_true")
    ap.add_argument("--no-mb-separators", dest="mb_separators", action="store_false")
    ap.set_defaults(mb_separators=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8008)
    ap.add_argument("--refresh-sec", type=float, default=2.0)
    ap.add_argument("--page-size", type=int, default=64)
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args(list(argv) if argv is not None else None)

    if args.mode == "serve":
        if args.tail_iters is None and args.iter_start is None and args.iter_end is None:
            args.tail_iters = 50
        return _serve_live_dashboard(args)

    if args.demo:
        payloads = _demo_payloads()
        tmp_dir = os.path.dirname(os.path.abspath(args.output)) or "."
        os.makedirs(tmp_dir, exist_ok=True)
        values: Dict[int, RankAverages] = {}
        world_size = 0
        for pl in payloads:
            rank = int(pl["rank"])
            world_size = max(world_size, int(pl["world_size"]))
            fwd, bwd = _select_records(pl, args.metric, args.iter_start, args.iter_end, args.mb_start, args.mb_end)
            values[rank] = RankAverages(rank=rank, forward_ms=_mean_or_nan(fwd), backward_ms=_mean_or_nan(bwd))
        avgs = [values[k] for k in sorted(values.keys())]
    else:
        payloads, _files = _load_payloads_from_inputs(args.input)
        args.iter_start, args.iter_end = _apply_tail_iters(args.iter_start, args.iter_end, payloads, args.tail_iters)

        avgs, world_size = build_rank_averages(
            inputs=args.input,
            metric=args.metric,
            iter_start=args.iter_start,
            iter_end=args.iter_end,
            mb_start=args.mb_start,
            mb_end=args.mb_end,
        )

    if not avgs:
        raise SystemExit("没有找到可用的 rank_*.json 或 records 为空")

    if args.mode in ("heatmap", "both"):
        png = _render_heatmap_png_bytes(
            avgs=avgs,
            world_size=world_size,
            unit=args.unit,
            cmap=args.cmap,
            annotate=bool(args.annotate),
            grid_rows=args.grid_rows,
            grid_cols=args.grid_cols,
            shared_scale=bool(args.shared_scale),
            metric=args.metric,
        )
        out_dir = os.path.dirname(os.path.abspath(args.output)) or "."
        os.makedirs(out_dir, exist_ok=True)
        with open(args.output, "wb") as f:
            f.write(png)

    if args.mode in ("timeline", "both"):
        f_map, b_map, ws = _build_rank_phase_maps(payloads, args.metric, args.iter_start, args.iter_end, args.mb_start, args.mb_end)
        _ = ws
        keys = _collect_keys(f_map, b_map)
        ranks = sorted(set(list(f_map.keys()) + list(b_map.keys())))
        if args.mode == "timeline":
            timeline_out = args.output
        else:
            base, ext = os.path.splitext(args.output)
            timeline_out = f"{base}_timeline{ext or '.png'}"
        png = _render_rank_mb_phase_png_bytes(
            forward_by_rank=f_map,
            backward_by_rank=b_map,
            ranks=ranks,
            keys=keys,
            unit=args.unit,
            annotate=bool(args.annotate),
            cmap=args.cmap,
            phase=str(args.timeline_phase),
            robust_scale=bool(args.timeline_robust_scale),
            scale_vmin=None,
            scale_vmax=None,
        )
        out_dir = os.path.dirname(os.path.abspath(timeline_out)) or "."
        os.makedirs(out_dir, exist_ok=True)
        with open(timeline_out, "wb") as f:
            f.write(png)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
