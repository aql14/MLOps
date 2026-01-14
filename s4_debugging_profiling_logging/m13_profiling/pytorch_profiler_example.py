import torch
import torchvision.models as models
from torch.profiler import profile, ProfilerActivity
from torch.profiler import profile, tensorboard_trace_handler

model = models.resnet18()
inputs = torch.randn(5, 3, 224, 224)

# Added profile_memory=True to track allocations. one time is unreliable
# with profile(activities=[ProfilerActivity.CPU], record_shapes=True, profile_memory=True) as prof:
#     model(inputs)

# Better to to it multiple times
# with profile(activities=[ProfilerActivity.CPU], record_shapes=True, profile_memory=True) as prof:
#     for i in range(10):
#         model(inputs)
#         prof.step()

# To visualize
with profile(
    activities=[ProfilerActivity.CPU],
    record_shapes=True,
    profile_memory=True,
    on_trace_ready=tensorboard_trace_handler("./log/resnet18") # Saves to this folder
) as prof:
    for i in range(10):
        model(inputs)
        prof.step() # Signals the profiler that an iteration has finished
        

# Sort by 'self_cpu_memory_usage' to find the biggest memory consumers
print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10))

# Export the profiling results
# prof.export_chrome_trace("trace.json")