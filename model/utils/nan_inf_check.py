import json
from datetime import datetime
import torch
from audiotools.ml.decorators import Tracker
import os

def _tensor_stats(x: torch.Tensor):
    # 只做轻量统计，避免巨大开销
    x = x.detach()
    return {
        "shape": list(x.shape),
        "dtype": str(x.dtype),
        "device": str(x.device),
        "min": float(torch.nan_to_num(x).min().item()) if x.numel() else None,
        "max": float(torch.nan_to_num(x).max().item()) if x.numel() else None,
        "mean": float(torch.nan_to_num(x).mean().item()) if x.numel() else None,
        "num_nan": int(torch.isnan(x).sum().item()),
        "num_inf": int(torch.isinf(x).sum().item()),
    }

def check_finite_or_fail(
    items: dict,
    *,
    step: int,
    accel,
    save_path: str,
    tracker: Tracker = None,
    extra: dict = None,
    dump_tensors: bool = True,
    raise_on_bad: bool = True,
):
    """
    items: {name: scalar/tensor}
    - 检测 NaN/Inf
    - rank0 追加写 anomaly.log
    - 可选 dump 一个 .pth 包含 batch/统计信息
    - raise 终止训练（推荐）
    """
    bad = []
    details = {}

    for k, v in items.items():
        if v is None:
            continue
        if torch.is_tensor(v):
            finite = torch.isfinite(v).all().item()
            if not finite:
                bad.append(k)
                details[k] = _tensor_stats(v)
        else:
            # python number
            try:
                fv = float(v)
                if not (fv == fv) or fv == float("inf") or fv == float("-inf"):
                    bad.append(k)
                    details[k] = {"value": v}
            except Exception:
                # 非法类型也记一下
                bad.append(k)
                details[k] = {"value_repr": repr(v)}

    if len(bad) == 0:
        return

    record = {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "step": int(step),
        "rank": int(getattr(accel, "local_rank", 0)),
        "world_size": int(getattr(accel, "world_size", 1)),
        "bad_keys": bad,
        "details": details,
        "extra": extra or {},
    }

    # rank0 写日志文件
    if int(getattr(accel, "local_rank", 0)) == 0:
        os.makedirs(save_path, exist_ok=True)
        log_file = os.path.join(save_path, "anomaly.log")
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # 可选：dump debug 包（建议只在异常时做）
    if dump_tensors:
        dump_obj = {
            "record": record,
        }
        dump_path = os.path.join(
            save_path, f"anomaly_step{int(step)}_rank{int(getattr(accel,'local_rank',0))}.pth"
        )
        try:
            torch.save(dump_obj, dump_path)
        except Exception as e:
            if tracker is not None:
                tracker.print(f"[WARN] Failed to torch.save anomaly dump: {e}")

    msg = f"[FATAL] NaN/Inf detected at step={step}, bad={bad}"
    if tracker is not None:
        tracker.print(msg)

    if raise_on_bad:
        raise FloatingPointError(msg)