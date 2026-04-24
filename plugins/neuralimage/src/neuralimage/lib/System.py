import torch


def check_gpu_availability():
    gpus = torch.cuda.device_count()
    return gpus

if __name__ == '__main__':
    print(check_gpu_availability())
    for i in range(torch.cuda.device_count()):
        device_name = torch.cuda.get_device_name(i)
        device_properties = torch.cuda.get_device_properties(i)
        print(f"Device {i}:")
        print(f"Name: {device_name}")
        print(f"Properties: {device_properties}\n")
    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.max_memory_allocated(i)
        cached = torch.cuda.max_memory_reserved(i)
        print(f"Device {i} Memory:")
        print(f"Allocated: {allocated / 1024 ** 3:.2f} GB")
        print(f"Cached: {cached / 1024 ** 3:.2f} GB\n")

