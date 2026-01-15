import atexit
import json
import os
import time
from typing import Callable, Optional, Sequence, Tuple

import torch
import torch.distributed as dist


class MicrobatchModuleHookProfiler:
    def __init__(self, module: torch.nn.Module, args_getter: Callable[[], object]):
        self._module = module
        self._get_args = args_getter
        self._enabled = False
        self._active = False
        self._handles = []
        self._last_seen_iter = None

        self._seq = 0
        self._fwd_mb_counter = {}
        self._bwd_mb_counter = {}
        self._mb_records = {}
        self._fwd_stack = []
        self._bwd_stack = []
        self._pending_mb_queue = {}
        self._bwd_inflight = {}

        self._grad_tensors = []

    def enable_if_configured(self) -> bool:
        args = self._get_args()
        if not getattr(args, "module_hook_profile", False):
            return False

        self._enabled = True
        self._active = True
        self._seq = 0
        self._last_seen_iter = None
        self._fwd_mb_counter = {}
        self._bwd_mb_counter = {}
        self._mb_records = {}
        self._fwd_stack = []
        self._bwd_stack = []
        self._pending_mb_queue = {}
        self._bwd_inflight = {}
        self._grad_tensors = [p for p in self._module.parameters() if getattr(p, "requires_grad", False)]

        try:
            self._handles.append(self._module.register_forward_pre_hook(self._on_forward_pre))
            self._handles.append(self._module.register_forward_hook(self._on_forward))
            self._handles.append(self._module.register_full_backward_pre_hook(self._on_backward_pre))
            self._handles.append(self._module.register_full_backward_hook(self._on_backward))
        except Exception:
            self._active = False
            return False

        try:
            atexit.register(self.flush, force=True)
        except Exception:
            pass

        try:
            self._init_output_file()
        except Exception:
            pass

        return True

    def flush(self, force: bool = False):
        if not self._enabled:
            return
        if not force and not self._active:
            return
        keys = list((self._mb_records or {}).keys())
        keys.sort(key=lambda k: (int(k[0]), int(k[1])))
        for it, mb in keys:
            self._flush_microbatch(int(it), int(mb))
        self._active = False

    def flush_iteration(self, iteration: int):
        if not self._enabled:
            return
        if not self._active:
            return
        keys = [k for k in (self._mb_records or {}).keys() if int(k[0]) == int(iteration)]
        keys.sort(key=lambda k: int(k[1]))
        for _, mb in keys:
            self._flush_microbatch(int(iteration), int(mb))

    def _should_record(self) -> bool:
        args = self._get_args()
        if hasattr(args, "_module_hook_profile_step_active") and not bool(getattr(args, "_module_hook_profile_step_active", False)):
            return False
        if not bool(getattr(self._module, "training", False)):
            return False
        curr_iteration = int(getattr(args, "curr_iteration", -1))
        start = int(getattr(args, "profile_step_start", 0))
        end = int(getattr(args, "profile_step_end", 0))
        if curr_iteration < start or curr_iteration >= end:
            return False
        return True

    def _on_forward_pre(self, module, inputs, *unused_args, **unused_kwargs):
        if not self._active:
            return
        args = self._get_args()
        curr_iter = int(getattr(args, "curr_iteration", -1))
        last_seen = self._last_seen_iter
        if last_seen is None:
            self._last_seen_iter = curr_iter
        elif int(last_seen) != curr_iter:
            self.flush_iteration(int(last_seen))
            self._last_seen_iter = curr_iter
            self._fwd_stack = []
            self._bwd_stack = []
            self._fwd_mb_counter = {}
            self._bwd_mb_counter = {}
            self._mb_records = {}
            self._pending_mb_queue = {}
            self._bwd_inflight = {}

        if not self._should_record():
            return
        self._seq += 1
        start_event = None
        if torch.cuda.is_available():
            start_event = torch.cuda.Event(enable_timing=True)
            start_event.record(torch.cuda.current_stream())
        current = int(self._fwd_mb_counter.get(int(curr_iter), 0))
        current += 1
        self._fwd_mb_counter[int(curr_iter)] = int(current)
        mb = int(current)
        self._fwd_stack.append(
            {
                "iter": int(curr_iter),
                "seq": int(self._seq),
                "mb": int(mb),
                "start_event": start_event,
                "cpu_start_s": float(time.time()),
            }
        )

    def _on_forward(self, module, inputs, outputs, *unused_args, **unused_kwargs):
        if not self._active:
            return
        if not self._should_record():
            return
        args = self._get_args()
        curr_iter = int(getattr(args, "curr_iteration", -1))
        if not self._fwd_stack:
            return
        st = self._fwd_stack[-1]
        if int(st.get("iter", -1)) != int(curr_iter):
            return
        end_event = None
        if torch.cuda.is_available():
            end_event = torch.cuda.Event(enable_timing=True)
            end_event.record(torch.cuda.current_stream())
        mb = int(st.get("mb", -1) or -1)
        mb_key = (int(curr_iter), int(mb))
        bucket = self._mb_records.get(mb_key, None)
        if bucket is None:
            bucket = {"forward": [], "backward": []}
            self._mb_records[mb_key] = bucket
        bucket["forward"].append(
            {
                "start_event": st.get("start_event", None),
                "end_event": end_event,
                "cpu_start_s": float(st.get("cpu_start_s", time.time())),
                "cpu_end_s": float(time.time()),
            }
        )
        pending = self._pending_mb_queue.get(int(curr_iter), None)
        if pending is None:
            pending = []
            self._pending_mb_queue[int(curr_iter)] = pending
        pending.append(int(mb))
        self._fwd_stack.pop()

    def _on_backward_pre(self, module, grad_output, *unused_args, **unused_kwargs):
        if not self._active:
            return
        if not self._should_record():
            return
        args = self._get_args()
        curr_iter = int(getattr(args, "curr_iteration", -1))
        last_seen = self._last_seen_iter
        if last_seen is None:
            self._last_seen_iter = curr_iter
        elif int(last_seen) != curr_iter:
            self.flush_iteration(int(last_seen))
            self._last_seen_iter = curr_iter
            self._fwd_stack = []
            self._bwd_stack = []
            self._fwd_mb_counter = {}
            self._bwd_mb_counter = {}
            self._mb_records = {}
            self._pending_mb_queue = {}
            self._bwd_inflight = {}

        self._seq += 1
        cpu_start_s = float(time.time())
        start_event = None
        if torch.cuda.is_available():
            start_event = torch.cuda.Event(enable_timing=True)
            start_event.record(torch.cuda.current_stream())
        pending = self._pending_mb_queue.get(int(curr_iter), None) or []
        mb = None
        if pending:
            mb = int(pending.pop(0))
            self._pending_mb_queue[int(curr_iter)] = pending
        if mb is None:
            current = int(self._bwd_mb_counter.get(int(curr_iter), 0))
            current += 1
            self._bwd_mb_counter[int(curr_iter)] = int(current)
            mb = int(current)
        mb_key = (int(curr_iter), int(mb))
        self._bwd_inflight[mb_key] = {
            "start_event": start_event,
            "cpu_start_s": cpu_start_s,
        }
        self._bwd_stack.append(
            {
                "iter": int(curr_iter),
                "seq": int(self._seq),
                "mb": int(mb),
            }
        )

        if not self._grad_tensors:
            end_event = None
            if torch.cuda.is_available():
                end_event = torch.cuda.Event(enable_timing=True)
                end_event.record(torch.cuda.current_stream())
            bucket = self._mb_records.get(mb_key, None)
            if bucket is None:
                bucket = {"forward": [], "backward": []}
                self._mb_records[mb_key] = bucket
            bucket["backward"].append(
                {
                    "start_event": start_event,
                    "end_event": end_event,
                    "cpu_start_s": float(cpu_start_s),
                    "cpu_end_s": float(time.time()),
                }
            )
            self._bwd_inflight.pop(mb_key, None)
            self._flush_microbatch(int(mb_key[0]), int(mb_key[1]))
            return

        try:
            import torch.autograd.graph as autograd_graph

            handle_box = {}

            def _on_all_grads_ready(_unused_grads, _key=mb_key):
                try:
                    h = handle_box.get("h", None)
                    if h is not None:
                        h.remove()
                except Exception:
                    pass
                if not self._active:
                    return
                info = self._bwd_inflight.pop(_key, None)
                if info is None:
                    return
                end_event = None
                if torch.cuda.is_available():
                    end_event = torch.cuda.Event(enable_timing=True)
                    end_event.record(torch.cuda.current_stream())
                bucket = self._mb_records.get(_key, None)
                if bucket is None:
                    bucket = {"forward": [], "backward": []}
                    self._mb_records[_key] = bucket
                bucket["backward"].append(
                    {
                        "start_event": info.get("start_event", None),
                        "end_event": end_event,
                        "cpu_start_s": float(info.get("cpu_start_s", time.time())),
                        "cpu_end_s": float(time.time()),
                    }
                )
                self._flush_microbatch(int(_key[0]), int(_key[1]))

            handle_box["h"] = autograd_graph.register_multi_grad_hook(self._grad_tensors, _on_all_grads_ready)
        except Exception:
            pass

    def _on_backward(self, module, grad_input, grad_output, *unused_args, **unused_kwargs):
        if not self._active:
            return
        if not self._should_record():
            return
        args = self._get_args()
        curr_iter = int(getattr(args, "curr_iteration", -1))
        if not self._bwd_stack:
            return
        st = self._bwd_stack[-1]
        if int(st.get("iter", -1)) != int(curr_iter):
            return
        self._bwd_stack.pop()

    def _flush_microbatch(self, iteration: int, microbatch: int):
        if not self._enabled or not self._active:
            return
        mb_key = (int(iteration), int(microbatch))
        bucket = (self._mb_records or {}).get(mb_key, None)
        if not bucket:
            return

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        forward_total_ms = 0.0
        backward_total_ms = 0.0

        for item in bucket.get("forward", []) or []:
            st = item.get("start_event", None)
            ed = item.get("end_event", None)
            if st is not None and ed is not None:
                try:
                    forward_total_ms += float(st.elapsed_time(ed))
                    continue
                except Exception:
                    pass
            try:
                forward_total_ms += (float(item["cpu_end_s"]) - float(item["cpu_start_s"])) * 1000.0
            except Exception:
                pass

        for item in bucket.get("backward", []) or []:
            st = item.get("start_event", None)
            ed = item.get("end_event", None)
            if st is not None and ed is not None:
                try:
                    backward_total_ms += float(st.elapsed_time(ed))
                    continue
                except Exception:
                    pass
            try:
                backward_total_ms += (float(item["cpu_end_s"]) - float(item["cpu_start_s"])) * 1000.0
            except Exception:
                pass

        payload, out_path = self._load_payload()
        payload["records"].extend(
            [
                {
                    "phase": "forward_total",
                    "iter": int(iteration),
                    "mb": int(microbatch),
                    "gpu_ms": float(forward_total_ms),
                    "cpu_ms": float(forward_total_ms),
                },
                {
                    "phase": "backward_total",
                    "iter": int(iteration),
                    "mb": int(microbatch),
                    "gpu_ms": float(backward_total_ms),
                    "cpu_ms": float(backward_total_ms),
                },
            ]
        )
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        try:
            del self._mb_records[mb_key]
        except Exception:
            pass

    def _get_rank_world_size(self) -> Tuple[int, int]:
        if dist.is_available() and dist.is_initialized():
            return int(dist.get_rank()), int(dist.get_world_size())
        rank_env_candidates = [
            "RANK",
            "SLURM_PROCID",
            "PMI_RANK",
            "OMPI_COMM_WORLD_RANK",
            "MPI_RANK",
            "LOCAL_RANK",
        ]
        world_size_env_candidates = [
            "WORLD_SIZE",
            "SLURM_NTASKS",
            "PMI_SIZE",
            "OMPI_COMM_WORLD_SIZE",
            "MPI_WORLD_SIZE",
        ]

        rank = 0
        for k in rank_env_candidates:
            v = os.environ.get(k, None)
            if v is None:
                continue
            try:
                rank = int(v)
                break
            except Exception:
                continue

        world_size = 1
        for k in world_size_env_candidates:
            v = os.environ.get(k, None)
            if v is None:
                continue
            try:
                world_size = int(v)
                break
            except Exception:
                continue
        return int(rank), int(world_size)

    def _get_out_path(self) -> str:
        args = self._get_args()
        base_dir = getattr(args, "module_hook_profile_path", None) or getattr(args, "profile_path", None) or "."
        out_dir = os.path.join(base_dir, "per_rank_json")
        os.makedirs(out_dir, exist_ok=True)
        rank, _ = self._get_rank_world_size()
        return os.path.join(out_dir, f"rank_{rank}.json")

    def _load_payload(self):
        args = self._get_args()
        rank, world_size = self._get_rank_world_size()
        out_path = self._get_out_path()
        payload = {
            "rank": rank,
            "world_size": world_size,
            "profile_step_start": int(getattr(args, "profile_step_start", 0)),
            "profile_step_end": int(getattr(args, "profile_step_end", 0)),
            "records": [],
        }
        if os.path.exists(out_path):
            try:
                with open(out_path, "r", encoding="utf-8") as f:
                    existing = json.load(f)
                if isinstance(existing, dict) and "records" in existing and isinstance(existing["records"], list):
                    payload = existing
            except Exception:
                pass
        return payload, out_path

    def _init_output_file(self):
        out_path = self._get_out_path()
        if os.path.exists(out_path):
            return
        payload, _ = self._load_payload()
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
