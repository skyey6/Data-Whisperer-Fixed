import torch
import os
import random
import numpy as np
import json
import time
import time
from functools import wraps

import logging
from typing import Any, Dict, List, Optional, Tuple
from colorlog import ColoredFormatter
from torch import distributed as dist

INVALID_INPUTS = {
    "bioinstruct": [
        "", "-", "n/a", "(n/a)", "none", "(none)", "(empty)", "(empty context)",
        "not applicable", "(not applicable)",
        "not required", "none required", "(no input required)", "(not required)", "(no specific input required)",
        "none needed", "(no input needed)",
        "(no specific input)", "(no specific input given)",
        "(no input)", "(no input provided)", "(no input necessary)",
    ],
    "dialogsum": [],  # 暂未发现dialogsum中存在类似情况
    "gsm8k": [],  # 暂无
}

logger = logging.getLogger('global')

def preprocess_function(examples, tokenizer, max_length=128):
    return {
        **tokenizer(
            examples["text"], truncation=True, padding="max_length", max_length=max_length
        ),
        "labels": examples["label"],
    }

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)

def load_json(file_path):
    """Load JSON file."""
    with open(file_path, "r") as f:
        return json.load(f)

def save_json(file_path, data):
    """Save data to JSON file."""
    with open(file_path, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def load_jsonl(file_path):
    with open(file_path, 'r') as f:
        return [json.loads(line) for line in f]

    
    
def save_args(args, output_dir, time, name):
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, f"{name}_{time}.json"), 'w') as f:
        json.dump(vars(args), f, indent=4)

from functools import wraps

def timer_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs) 
        end_time = time.time() 
        execution_time = end_time - start_time
        logger.info(f"Function '{func.__name__}' executed in {execution_time:.4f} seconds.")
        return result

    return wrapper


def process_val_samples(val_samples):
    """Process validation samples to remove answers (content after ####)."""
    processed_samples = []
    for sample in val_samples:
        processed_sample = sample.copy()
        # Remove content after '####' in the answer
        if 'answer' in processed_sample and '####' in processed_sample['answer']:
            processed_sample['answer'] = processed_sample['answer'].split('####')[0].strip()
        processed_samples.append(processed_sample)
    return processed_samples


def load_json_or_jsonl(file_path: str) -> List[Dict[str, Any]]:
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read().strip()
    if not text:
        return []
    if text[0] == "[":
        return json.loads(text)
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def setup_logger(only_rank0=True, is_debug=False):
    """Setup logger so that only rank0 prints logs (safe for DDP/FSDP)."""
    rank = 0
    # check distributed environment and get rank
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        log_format = f"[%(asctime)s (rank{rank})] - %(log_color)s%(levelname)s%(reset)s - %(filename)s:%(lineno)s - %(message)s"
    else:
        log_format = "[%(asctime)s] - %(log_color)s%(levelname)s%(reset)s - %(filename)s:%(lineno)s - %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    logger = logging.getLogger('global') # create logger

    if logger.handlers:
        return logger

    if only_rank0 and rank != 0:
        # non-rank0: mute logger
        logger.addHandler(logging.NullHandler())
        return logger

    handler = logging.StreamHandler()

    formatter = ColoredFormatter(
            fmt=log_format,
            datefmt=date_format,
            log_colors={
                "DEBUG": "cyan",
                "INFO": "green",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "bold_red",
            },
    )
    # formatter = logging.Formatter(fmt=log_format, datefmt=date_format)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG if is_debug else logging.INFO)
    logger.propagate = False

    return logger


def build_prompt(dataset_name: str, instruction: str, input: Optional[str] = None) -> str:
    """Build prompt for a single data sample."""
    has_valid_input = input and input.strip().rstrip('.').casefold() not in INVALID_INPUTS[dataset_name]
    if has_valid_input:
        return f'Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.' \
             + f'\n\n### Instruction:\n{instruction}' \
             + f'\n\n### Input:\n{input}' \
             + f'\n\n### Response:\n'
    else:
        return f'Below is an instruction that describes a task. Write a response that appropriately completes the request.' \
             + f'\n\n### Instruction:\n{instruction}' \
             + f'\n\n### Response:\n'


def save_json(file_path: str, data: Any) -> None:
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def save_jsonl(file_path: str, records: List[Dict[str, Any]]) -> None:
    with open(file_path, "w", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")