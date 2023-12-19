import torch
import time

class Timer:
    def __init__(self):
        self.start = time.time()
    def duration(self, rd=3):
        return round(time.time() - self.start, rd)
    def reset(self):
        self.start = time.time()
        
class CudaTimer:
    def __init__(self, stream=torch.cuda.current_stream()):
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)
        self.stream = stream
        self.end_recorded = False
        self.start_event.record(stream=self.stream)

    def start(self):
        self.start_event.record(stream=self.stream)
        
    def end(self):
        self.end_event.record(stream=self.stream)
        self.end_recorded = True
        
    def reset(self):
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)
        self.end_recorded = False
        
    def duration(self):
        assert(self.end_recorded)
        self.start_event.synchronize()
        self.end_event.synchronize()
        duration_ms = self.start_event.elapsed_time(self.end_event)
        duration_s = duration_ms / 1000
        return duration_s