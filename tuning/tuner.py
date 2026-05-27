import os
import sys
import math
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
from dataclasses import dataclass, field
from datasets import load_dataset, Dataset
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    HfArgumentParser,
    Trainer,
    DataCollatorForSeq2Seq,
)
# Add project root to sys.path for imports
BASE_PATH = Path(os.path.abspath(__file__)).parent.parent
sys.path.append(str(BASE_PATH))
from peft import LoraConfig, get_peft_model
from utils.utils import *


def build_dataset_prompt(dataset_name: str, sample: Dict[str, Any]) -> str:
    if dataset_name == "bioinstruct":
        return build_prompt(dataset_name, sample["instruction"], sample["input"])
    if dataset_name == "dialogsum":
        return (
            "Below is a dialogue. Write a clear, concise, and complete summary for the dialogue."
            "\n\n### Guidelines:"
            "\n1. Capture the main points and important outcomes of the dialogue."
            "\n2. Do not add information that is not mentioned in the dialogue."
            "\n3. Do not include extra commentary, explanations, or formatting outside the summary itself."
            f'\n\n### Dialogue:\n{sample["dialogue"]}'
            "\n\n### Response:\n"
        )
    if dataset_name == "gsm8k":
        return (
            "Below is a math word problem. Solve it step by step, and then give the final answer in the exact required format."
            "\n\n### Guidelines:"
            "\n1. Your response should contain the reasoning steps needed to solve the problem."
            "\n2. Keep the reasoning clear, concise, and logically correct."
            "\n3. Do not add extra commentary beyond the solution."
            "\n4. The final line must be: #### <number>"
            "\n5. Replace <number> with the final numeric answer only."
            f'\n\n### Question:\n{sample["question"]}'
            "\n\n### Response:\n"
        )
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def get_output_field(dataset_name: str) -> str:
    if dataset_name == "bioinstruct":
        return "output"
    if dataset_name == "dialogsum":
        return "summary"
    if dataset_name == "gsm8k":
        return "answer"
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def init_distributed() -> Tuple[int, int, bool]:
    """Initialize distributed environment if available."""
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    is_distributed = local_rank >= 0 and world_size > 1
    if is_distributed:
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        if dist.is_available() and not dist.is_initialized():
            dist.init_process_group(backend="nccl")
            logger.info(f"Initialized distributed environment with {local_rank=} and {world_size=}.")
    
    return local_rank, world_size, is_distributed


@timer_decorator
def load_data(data_path, selection_rate: float = 1.0) -> Dataset:
    # TODO: 读取数据后score的精度会变差，可能是load_dataset的json解析导致的
    # 后续如果有必要可以改成直接读取json文件来加载数据
    ds = load_dataset("json", data_files=data_path, split="train")
    logger.info(f"Loaded dataset with {len(ds)} samples")
    # Apply selection rate
    assert 0 < selection_rate <= 1.0, "selection_rate must be in (0, 1]"
    if selection_rate < 1.0:
        num_samples = math.floor(len(ds) * selection_rate)
        ds = ds.select(range(num_samples))
    return ds


@timer_decorator
def load_model(model_path, use_lora=True):
    """Load model and tokenizer""" 
    # load model
    model = AutoModelForCausalLM.from_pretrained(model_path)
    if use_lora:
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        )
        model = get_peft_model(model, lora_config)
        if not dist.is_initialized() or int(os.environ.get("LOCAL_RANK", -1)) == 0:
            print("LoRA enabled:", end="\n\t")
            model.print_trainable_parameters()
    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    # HF tokenizer bug fix
    has_pad_token = getattr(tokenizer, "pad_token", False)
    has_eos_token = getattr(tokenizer, "eos_token", False)
    if not has_pad_token:
        if has_eos_token:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            raise ValueError("The tokenizer does not have a pad_token or eos_token. Please check the tokenizer configuration.")
    
    return model, tokenizer


@dataclass
class ScriptArguments:
    dataset_name: Optional[str] = field(default=None, init=False)
    target_dir: str
    data_path: Optional[str] = field(default=None, init=False)
    selection_rate: float
    model_name: Optional[str]
    model_path: Optional[str] = field(default=None, init=False)
    output_dir: Optional[str] = field(default=None, init=False)
    # ---- train ----
    num_train_epochs: float
    learning_rate: float
    per_device_train_batch_size: int
    lr_scheduler_type: str = "cosine"
    use_lora: bool = True
    # ----  log  ----
    log_level: str = "info"
    log_level_replica: str = "warning"
    logging_steps: int = 100
    logging_first_step: bool = True

    def __post_init__(self):
        self.dataset_name = self.target_dir.split("_")[1]
        self.data_path = str(BASE_PATH / "results" / "pruning" / self.target_dir / "data_whisperer.json")
        if "random" in self.target_dir.lower() and not self.model_name:
            raise ValueError(
                "For random selector runs, model_name must be specified to determine the model_path of the pre-trained model."
            )
        if "random" not in self.target_dir.lower():
            self.model_name = self.target_dir.split("_", 1)[0]
            self.output_dir = str(BASE_PATH / "results" / "tuning" / self.target_dir / f"sr{self.selection_rate}")
        else:
            self.output_dir = str(BASE_PATH / "results" / "tuning" / self.target_dir / self.model_name / f"sr{self.selection_rate}")
        self.model_path = str(BASE_PATH / "model" / self.model_name)


if __name__ == "__main__":
    debug_mode = True
    parser = HfArgumentParser(ScriptArguments)
    args = parser.parse_args_into_dataclasses()[0]
    assert args.target_dir, "Need to specify target_dir to distinguish different tuning runs, e.g., 'Llama-3-8B-Instruct_bioinstruct_datawhisperer_rouge-L_10_5_5'"
    # pprint(args)

    local_rank, world_size, is_distributed = init_distributed()
    # initialize logger after distributed setup
    logger = setup_logger(only_rank0=True, is_debug=debug_mode)
    logger.info(f"Training arguments:\n{args}")

    # load dataset and model/tokenizer
    raw_dataset = load_data(args.data_path, args.selection_rate)
    model, tokenizer = load_model(args.model_path, use_lora=args.use_lora)
    model_max_length = model.config.max_position_embeddings  # 在preprocess_data中使用

    def preprocess_data(examples):
        max_len = model_max_length
        prompts = [
            build_dataset_prompt(args.dataset_name, sample)
            for sample in (dict(zip(examples.keys(), row)) for row in zip(*examples.values()))
        ]
        outputs = examples[get_output_field(args.dataset_name)]
        # tokenize
        prompt_ids = tokenizer(prompts, add_special_tokens=False)["input_ids"]
        output_ids = tokenizer(outputs, add_special_tokens=False)["input_ids"]
        input_ids_list = [
            ([tokenizer.bos_token_id] + p_ids + o_ids + [tokenizer.eos_token_id])[:max_len]
                for p_ids, o_ids in zip(prompt_ids, output_ids)
        ]
        # construct labels with -100 for prompt tokens and actual token ids for output tokens
        labels_list = [
            ([-100] * (1 + len(p_ids)) + o_ids + [tokenizer.eos_token_id])[:max_len]
                for p_ids, o_ids in zip(prompt_ids, output_ids)
        ]

        return {
            "input_ids": input_ids_list,
            "labels": labels_list,
            "attention_mask": [[1] * len(ids) for ids in input_ids_list]
        }

    # _run_preprocess_tests()
    tokenized_dataset = raw_dataset.map(
        preprocess_data, batched=True, remove_columns=raw_dataset.column_names
    )#.select(range(200))  # 选200条测试流程，正式运行时去掉 .select()
    # _test_collator_padding()

    # collator bug FIXED
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, padding=True, pad_to_multiple_of=8, return_tensors="pt"
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine", # linear warm-up and cosine decay
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=1,
        fp16=True,
        log_level=args.log_level,
        log_level_replica=args.log_level_replica,
        logging_steps=args.logging_steps,
        logging_first_step=args.logging_first_step,
        save_strategy="no",
        seed=42,
        label_names=["labels"],
        ddp_find_unused_parameters=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    train_result = trainer.train()
    logger.info(f"Training result:\n{train_result._asdict()}")
    # 只在主进程保存
    if trainer.is_world_process_zero():
        save_json(os.path.join(args.output_dir, "train_result.json"), train_result._asdict())
        trainer.model.save_pretrained(args.output_dir)
        logger.info(f"Training finished. Model saved to: {args.output_dir}")

    # 分布式资源清理
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
