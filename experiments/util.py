import subprocess
import platform
import re
import time
import os
    
def print_colored_block(message, ok=True, rows=6, row_length=80):
    col = '\033[92m' if ok else '\033[91m'
    print(col + (u"\u2588"*row_length + '\n')*rows + '\n')
    print(f"{message}\n")
    print((u"\u2588"*row_length + '\n')*rows + '\033[0m')

def get_processor_name():
    if platform.system() == "Windows":
        return platform.processor()
    elif platform.system() == "Linux":
        command = "cat /proc/cpuinfo"
        all_info = subprocess.check_output(command, shell=True).strip().decode('ascii')
        for line in all_info.split("\n"):
            if "model name" in line:
                return re.sub( ".*model name.*:", "", line,1).strip()
    return ""

def get_gpu_name():
    system = platform.system()
    try:
        if system == "Linux":
            # NVIDIA GPUs
            return subprocess.check_output(
                ["nvidia-smi", "--query-gpu=gpu_name", "--format=csv,noheader"],
                stderr=subprocess.DEVNULL
            ).decode().strip().split("\n")[0]

        elif system == "Darwin":  # macOS
            output = subprocess.check_output(
                ["system_profiler", "SPDisplaysDataType"],
                stderr=subprocess.DEVNULL
            ).decode()
            for line in output.split("\n"):
                if "Chipset Model" in line:
                    return line.split(":")[1].strip()

        elif system == "Windows":
            output = subprocess.check_output(
                ["wmic", "path", "win32_videocontroller", "get", "name"],
                stderr=subprocess.DEVNULL
            ).decode().split("\n")
            gpus = [line.strip() for line in output if line.strip() and "Name" not in line]
            return gpus[0] if gpus else None

    except Exception:
        return None

    return None

def infos_from_dataset_path(path):
    file = os.path.basename(path)
    dataset, rest = split_next(file, "_split")
    split, rest = split_next(rest, "_n")
    n, rest = split_next(rest, "_seed")
    seed, rest = split_next(rest, "_model")
    model, rest = split_next(rest, "_temperature")
    temperature, rest = split_next(rest, "_flags")
    flags, rest = split_next(rest, ".jsonl")
    return {
        "dataset_path": path,
        "dataset_filename": file,
        "dataset": dataset,
        "dataset_split": split,
        "dataset_n_samples": n,
        "dataset_temperature": temperature,
        "dataset_flags": flags
    }

def split_next(rest, separator):
    rest = rest.split(separator, 1)
    return (rest[0], rest[1])
