import time, psutil
try:
    import pynvml
    pynvml.nvmlInit()
    _GPU = True
except Exception:
    _GPU = False

class CostTracker:
    def __enter__(self):
        self.t0 = time.time()
        self.cpu0 = psutil.Process().cpu_times()
        self.mem0 = psutil.Process().memory_info().rss
        if _GPU:
            h = pynvml.nvmlDeviceGetHandleByIndex(0)
            self.vram0 = pynvml.nvmlDeviceGetMemoryInfo(h).used
        return self

    def __exit__(self, a,b,c):
        self.t1 = time.time()
        self.cpu1 = psutil.Process().cpu_times()
        self.mem1 = psutil.Process().memory_info().rss
        if _GPU:
            h = pynvml.nvmlDeviceGetHandleByIndex(0)
            self.vram1 = pynvml.nvmlDeviceGetMemoryInfo(h).used

    def report(self):
        out = {
            "wall_s": self.t1 - self.t0,
            "cpu_user_s": self.cpu1.user - self.cpu0.user,
            "rss_mb": (self.mem1 - self.mem0)/1024/1024
        }
        if _GPU:
            out["vram_mb"] = (self.vram1 - self.vram0)/1024/1024
        return out
