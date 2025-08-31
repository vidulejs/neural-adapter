import torch

def clear_memory():

    import gc

    def _mb(x): return int(x / (1024**2))

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        print(f"CUDA before -> allocated: {_mb(torch.cuda.memory_allocated())} MB, "
            f"reserved: {_mb(torch.cuda.memory_reserved())} MB")

    # Move models to CPU and mark objects for deletion
    to_del = []

    for name in ["model", "optimizer", "scheduler", "x", "y", "y_pred", "current_x", "sample"]:
        if name in globals():
            obj = globals()[name]
            try:
                if isinstance(obj, nn.Module):
                    obj.to("cpu")
            except Exception:
                pass
            to_del.append(name)

    # Also scan globals for any CUDA tensors/modules/optimizers
    for name, obj in list(globals().items()):
        try:
            if torch.is_tensor(obj) and obj.is_cuda:
                to_del.append(name)
            elif isinstance(obj, nn.Module) and any(p.is_cuda for p in obj.parameters()):
                try:
                    obj.to("cpu")
                except Exception:
                    pass
                to_del.append(name)
            elif isinstance(obj, torch.optim.Optimizer):
                to_del.append(name)
        except Exception:
            pass

    for name in sorted(set(to_del)):
        try:
            del globals()[name]
        except Exception:
            pass

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass
        torch.cuda.synchronize()
        print(f"CUDA after  -> allocated: {_mb(torch.cuda.memory_allocated())} MB, "
            f"reserved: {_mb(torch.cuda.memory_reserved())} MB")

    # Optional: report remaining live CUDA tensors
    live_cuda = 0
    for o in gc.get_objects():
        try:
            if torch.is_tensor(o) and o.is_cuda:
                live_cuda += 1
        except Exception:
            pass
    print(f"Remaining live CUDA tensors: {live_cuda}")