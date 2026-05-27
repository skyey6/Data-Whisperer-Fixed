import os
import sys
import re
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser
from peft import PeftModel
BASE_PATH = Path(os.path.abspath(__file__)).parent.parent
sys.path.append(str(BASE_PATH))
from metrics.metric import METRICS
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


def format_sample(sample: Dict[str, Any], dataset_name: str) -> Tuple[str, str]:
    if dataset_name == "bioinstruct":
        return build_dataset_prompt(dataset_name, sample), sample["output"]
    if dataset_name == "dialogsum":
        return build_dataset_prompt(dataset_name, sample), sample["summary"]
    if dataset_name == "gsm8k":
        return build_dataset_prompt(dataset_name, sample), sample["answer"]
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def extract_gsm8k_final_answer(text: str) -> str:
    match = re.findall(r"####\s*([-+]?\d[\d,]*(?:\.\d+)?)", text)
    if match:
        return match[-1].replace(",", "").strip()

    numbers = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", text)
    if numbers:
        return numbers[-1].replace(",", "").strip()
    return text.strip()


def create_batch(dataset, batch_size) -> List[Dict]:
    batches = []
    num_batch = len(dataset)//batch_size if len(dataset) % batch_size == 0 else len(dataset)//batch_size + 1
    for i in range(num_batch):
        batch = dataset[i*batch_size: min((i+1)*batch_size, len(dataset))]
        batches.append(batch)
    return batches


@timer_decorator
def load_model(tuned_model_path: str, use_lora: bool, base_model_path: Optional[str] = None):
    if use_lora and not base_model_path:
        raise ValueError("use_lora is True but base_model_path is not provided.")
    # load model
    if use_lora:
        logger.info(f"Loading base model from: {base_model_path}")
        base_model = AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype=torch.float16)
        logger.info(f"Loading LoRA adapter from: {tuned_model_path}")
        model = PeftModel.from_pretrained(base_model, tuned_model_path)
    else:
        logger.info(f"Loading full model from: {tuned_model_path}")
        model = AutoModelForCausalLM.from_pretrained(tuned_model_path, torch_dtype=torch.float16)
    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, padding_side="left", add_eos_token=False)
    # HF tokenizer bug fix
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is None:
            raise ValueError("Tokenizer has neither pad_token nor eos_token.")
        tokenizer.pad_token = tokenizer.eos_token

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    logger.info(f"Model loaded on device: {device}")
    return model, tokenizer


def batched_predict(
    prompts: List[Dict[str, Any]],
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    max_input_length: int,
    max_new_tokens: int,
    **gen_kwargs
) -> List[str]:
    """Generate the output for a single example."""
    encodings = tokenizer(
        prompts, return_tensors='pt', truncation=True, max_length=max_input_length, padding='longest'
    ).to(model.device)
    with torch.no_grad():
        generation_output = model.generate(
            **encodings,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
            temperature=None,   # disable warning
            top_p=None,         # disable warning
            **gen_kwargs
        )
    # logger.debug(f"Generation output:\n{generation_output}")
    outputs = tokenizer.batch_decode(generation_output, skip_special_tokens=True)
    # logger.debug(f"Decoded output:\n{output}")
    predictions = []
    for output in outputs:
        if "### Response:" in output:
            predictions.append(output.split("### Response:", 1)[1].strip())
        else:
            predictions.append(output.strip())
    return predictions


@dataclass
class ScriptArguments:
    dataset_name: Optional[str] = field(default=None, init=False)
    test_path: Optional[str] = field(default=None, init=False)
    selection_rate: float
    target_dir: str
    base_model_name: Optional[str] = None
    base_model_path: Optional[str] = field(default=None, init=False)
    tuned_model_path: Optional[str] = field(default=None, init=False)
    output_dir: Optional[str] = field(default=None, init=False)
    metric: str = "auto"  # auto | rouge-L | exact_match
    use_lora: bool = True
    max_input_length: Optional[int] = field(default=None, init=False)
    max_new_tokens: Optional[int] = field(default=None, init=False)
    limit: int = -1  # limit the number of test samples for quick evaluation, -1 means no limit
    batch_size: int = 32

    def __post_init__(self):
        self.dataset_name = self.target_dir.split("_")[1]
        self.test_path = str(BASE_PATH / "data" / self.dataset_name / "test.json")
        if "random" in self.target_dir.lower() and not self.base_model_name:
            raise ValueError("For random selector runs, base_model_name must be specified to determine the base_model_path.")
        if "random" not in self.target_dir.lower():
            self.base_model_name = self.target_dir.split("_", 1)[0]
            self.tuned_model_path = str(BASE_PATH / "results" / "tuning" / self.target_dir / f"sr{self.selection_rate}")
        else:
            self.tuned_model_path = str(BASE_PATH / "results" / "tuning" / self.target_dir / self.base_model_name / f"sr{self.selection_rate}")
        self.base_model_path = str(BASE_PATH / "model" / self.base_model_name)
        self.output_dir = self.tuned_model_path
        if self.dataset_name == "bioinstruct":
            self.max_input_length = 2048
            self.max_new_tokens = 2048
        elif self.dataset_name == "dialogsum":
            self.max_input_length = 4096
            self.max_new_tokens = 512
        elif self.dataset_name == "gsm8k":
            self.max_input_length = 1024
            self.max_new_tokens = 1024
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")


def main(args: ScriptArguments) -> None:
    debug_mode = True
    logger = setup_logger(only_rank0=True, is_debug=debug_mode)
    # evaluate after tuning
    if not os.path.exists(args.output_dir):
        raise ValueError(f"Output directory does not exist: {args.output_dir}")

    logger.info(f"Evaluation arguments:\n{args}")

    test_set = load_json_or_jsonl(args.test_path)
    if args.limit > 0:
        test_set = test_set[: args.limit]
    logger.info(f"Loaded test set: {args.test_path}, size={len(test_set)}")
    data_batches = create_batch(test_set, args.batch_size)

    if args.metric == "auto":
        metric = "exact_match" if args.dataset_name == "gsm8k" else "rouge-L"
    else:
        metric = args.metric
    logger.info(f"Evaluating dataset: {args.dataset_name}, metric: {metric}")
    metric_function = METRICS[metric]

    model, tokenizer = load_model(
        args.tuned_model_path, use_lora=args.use_lora, base_model_path=args.base_model_path
    )

    all_scores: List[Optional[float]] = []
    all_records: List[Dict[str, Any]] = []
    # ==== start evaluation ====
    pbar = tqdm(total=len(test_set), desc="Evaluating")
    for i, batch in enumerate(data_batches):
        prompts, references = zip(*[format_sample(sample, args.dataset_name) for sample in batch])
        predictions = batched_predict(
            prompts, model, tokenizer, args.max_input_length, args.max_new_tokens
        )
        for j, (prediction, reference) in enumerate(zip(predictions, references)):
            if metric == "rouge-L":
                score = metric_function(reference, prediction)
            else:  # exact_match (gsm8k)
                formatted_reference = extract_gsm8k_final_answer(reference)
                formatted_prediction = extract_gsm8k_final_answer(prediction)
                score = metric_function(formatted_reference, formatted_prediction)  # boolean score
            record = {
                "idx": j + i * args.batch_size,
                "prediction": prediction,
                "reference": reference,
                "score": score,
            }
            if args.dataset_name == "gsm8k":
                record["prediction_final_answer"] = formatted_prediction
                record["reference_final_answer"] = formatted_reference
            all_scores.append(score)
            all_records.append(record)
        pbar.update(len(batch))
        pbar.set_postfix({"avg_score": float(sum(all_scores)) / len(all_scores)})

    avg_score = float(sum(all_scores) / max(1, len(all_scores)))
    summary = {
        "eval_config": asdict(args),
        "num_samples": len(all_scores),
        "average_score": avg_score,
    }
    summary["metric"] = metric

    pred_path = os.path.join(args.output_dir, "test_predictions.json")
    metric_path = os.path.join(args.output_dir, "test_summary.json")
    save_json(pred_path, all_records)
    save_json(metric_path, summary)

    logger.info(f"-------- Evaluation finished. --------")
    logger.info(f"Average {metric} = {avg_score:.6f}")
    logger.info(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    args = parser.parse_args_into_dataclasses()[0]
    main(args)
