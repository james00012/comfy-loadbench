"""Decompose the volume-to-VRAM load path on a real GPU.

Prints LOADBENCH lines the job log captures. Each measurement is guarded so one
failure cannot take the job down with it.
"""
import os
import time
import threading

MB = 1024 * 1024


def _say(label, nbytes, secs, extra=""):
    rate = (nbytes / MB) / secs if secs > 0 else 0.0
    print("LOADBENCH %-22s bytes=%-13d secs=%7.3f MBps=%8.1f %s"
          % (label, nbytes, secs, rate, extra), flush=True)
    return rate


def _resolve(path):
    """The checkpoint's real location, asking ComfyUI first."""
    try:
        import folder_paths
        hit = folder_paths.get_full_path("checkpoints", os.path.basename(path))
        if hit and os.path.exists(hit):
            return hit
    except Exception:
        pass
    return path


def _seq_read(path, chunk, limit, offset=0):
    got, t0 = 0, time.perf_counter()
    fd = os.open(path, os.O_RDONLY)
    try:
        while got < limit:
            b = os.pread(fd, min(chunk, limit - got), offset + got)
            if not b:
                break
            got += len(b)
    finally:
        os.close(fd)
    return got, time.perf_counter() - t0


def _threaded_read(path, nthreads, chunk, limit, offset=0):
    per = limit // nthreads
    counts = [0] * nthreads

    def work(i):
        start = offset + i * per
        end = offset + (limit if i == nthreads - 1 else (i + 1) * per)
        fd = os.open(path, os.O_RDONLY)
        try:
            off, tot = start, 0
            while off < end:
                b = os.pread(fd, min(chunk, end - off), off)
                if not b:
                    break
                off += len(b)
                tot += len(b)
            counts[i] = tot
        finally:
            os.close(fd)

    ts = [threading.Thread(target=work, args=(i,)) for i in range(nthreads)]
    t0 = time.perf_counter()
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    return sum(counts), time.perf_counter() - t0


class LoadBench:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "ckpt": ("STRING", {"default": "flux1-schnell-fp8.safetensors"}),
            "threads": ("INT", {"default": 8, "min": 1, "max": 64}),
            "chunk_mb": ("INT", {"default": 32, "min": 1, "max": 512}),
            "limit_gb": ("INT", {"default": 4, "min": 1, "max": 64}),
            "offset_gb": ("INT", {"default": 0, "min": 0, "max": 60}),
            "mode": (["seq", "threaded", "transfer", "library"],),
        }}

    RETURN_TYPES = ("STRING",)
    FUNCTION = "run"
    CATEGORY = "bench"
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, **kw):
        return time.time()

    def run(self, ckpt, threads, chunk_mb, limit_gb, offset_gb=0, mode="seq"):
        path = _resolve(ckpt)
        chunk = chunk_mb * MB
        limit = limit_gb * 1024 * MB
        offset = offset_gb * 1024 * MB
        out = ["path=%s" % path]
        print("LOADBENCH ==== start path=%s exists=%s size=%s"
              % (path, os.path.exists(path),
                 os.path.getsize(path) if os.path.exists(path) else -1), flush=True)

        if not os.path.exists(path):
            print("LOADBENCH ABORT file not found", flush=True)
            return ("missing",)
        limit = min(limit, os.path.getsize(path) - offset)
        print("LOADBENCH mode=%s offset=%d limit=%d threads=%d"
              % (mode, offset, limit, threads), flush=True)

        # One storage mode per job, at an offset nothing has read yet, so the
        # page cache cannot flatter the second measurement.
        if mode == "seq":
            try:
                n, s = _seq_read(path, chunk, limit, offset)
                out.append("seq=%.1f" % _say("read_seq", n, s, "threads=1 off=%d" % offset))
            except Exception as e:
                print("LOADBENCH read_seq FAILED %r" % (e,), flush=True)
        if mode == "threaded":
            try:
                n, s = _threaded_read(path, threads, chunk, limit, offset)
                out.append("threaded=%.1f" % _say("read_threaded", n, s,
                           "threads=%d off=%d" % (threads, offset)))
            except Exception as e:
                print("LOADBENCH read_threaded FAILED %r" % (e,), flush=True)
        if mode not in ("transfer", "library"):
            print("LOADBENCH ==== done", flush=True)
            return ("\n".join(out),)

        # Transfer: host to device, pageable against pinned.
        try:
            if mode != "transfer":
                raise StopIteration
            import torch
            if not torch.cuda.is_available():
                print("LOADBENCH no cuda", flush=True)
                return ("\n".join(out),)
            n = min(limit, 2 * 1024 * MB)
            print("LOADBENCH cuda=%s" % torch.cuda.get_device_name(0), flush=True)

            host = torch.empty(n, dtype=torch.uint8)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            host.to("cuda", non_blocking=False)
            torch.cuda.synchronize()
            out.append("h2d_pageable=%.1f" % _say("h2d_pageable", n, time.perf_counter() - t0))
            del host
            torch.cuda.empty_cache()

            pinned = torch.empty(n, dtype=torch.uint8).pin_memory()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            pinned.to("cuda", non_blocking=True)
            torch.cuda.synchronize()
            out.append("h2d_pinned=%.1f" % _say("h2d_pinned", n, time.perf_counter() - t0))
            del pinned
            torch.cuda.empty_cache()

            # The cast ComfyUI's own log names: fp8 weights, bfloat16 compute.
            try:
                elems = n // 2
                src = torch.empty(elems, dtype=torch.float8_e4m3fn, device="cuda")
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                src.to(torch.bfloat16)
                torch.cuda.synchronize()
                _say("fp8_to_bf16_on_gpu", elems, time.perf_counter() - t0)
                del src
                torch.cuda.empty_cache()
            except Exception as e:
                print("LOADBENCH cast FAILED %r" % (e,), flush=True)
        except StopIteration:
            pass
        except Exception as e:
            print("LOADBENCH transfer FAILED %r" % (e,), flush=True)

        # The library path, for reference against the raw numbers above.
        for dev in (("cpu", "cuda") if mode == "library" else ()):
            try:
                from safetensors.torch import load_file
                t0 = time.perf_counter()
                sd = load_file(path, device=dev)
                s = time.perf_counter() - t0
                tot = sum(v.numel() * v.element_size() for v in sd.values())
                _say("safetensors_%s" % dev, tot, s)
                del sd
                import torch as _t
                _t.cuda.empty_cache()
            except Exception as e:
                print("LOADBENCH safetensors_%s FAILED %r" % (dev, e), flush=True)

        print("LOADBENCH ==== done", flush=True)
        return ("\n".join(out),)


NODE_CLASS_MAPPINGS = {"LoadBench": LoadBench}
NODE_DISPLAY_NAME_MAPPINGS = {"LoadBench": "Load Benchmark"}
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
