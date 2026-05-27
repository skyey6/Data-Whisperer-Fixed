from datawhisperer_bioinstruct_pruner import DataWhisperer_BioInstruct_Pruner
from datawhisperer_dialog_pruner import DataWhisperer_Dialog_Pruner
from datawhisperer_gsm_pruner import DataWhisperer_GSM_Pruner
# from datawhisperer_qwen2_5_vl_pruner import DataWhisperer_Qwen2_5VL_Pruner
import argparse
import time
import torch
import threading
import os
import json

def get_pruner(dataset, method='datawhisperer'):
    if  method == 'datawhisperer':
        pruner_map = {
            "bioinstruct": DataWhisperer_BioInstruct_Pruner,
            "dialogsum": DataWhisperer_Dialog_Pruner,
            "gsm8k": DataWhisperer_GSM_Pruner,
            # "llava_1k": DataWhisperer_Qwen2_5VL_Pruner,
        }
    return pruner_map.get(dataset)

def monitor_cuda_memory(stop_event, peak_memory_dict, device_index=0):
    device = torch.device(f"cuda:{device_index}")
    peak_memory_allocated = 0
    peak_memory_reserved = 0
    while not stop_event.is_set():
        current_memory_allocated = torch.cuda.memory_allocated(device) / 1024**2 
        current_memory_reserved = torch.cuda.memory_reserved(device) / 1024**2
        if current_memory_allocated > peak_memory_allocated:
            peak_memory_allocated = current_memory_allocated
        if current_memory_reserved > peak_memory_reserved:
            peak_memory_reserved = current_memory_reserved
        time.sleep(0.1)  
    peak_memory_dict['memory_allocated'] = peak_memory_allocated
    peak_memory_dict['memory_reserved'] = peak_memory_reserved

def run_pruning(args):
    Pruner = get_pruner(args.dataset, args.method)
    pruner = Pruner(args)
    pruner.do_pruning()

if __name__ == "__main__":

    # Benhao: uncomment this to debug the code
    # import debugpy
    # debugpy.listen(5679)
    # print("Waiting for debugger attach...")
    # debugpy.wait_for_client()

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to the pretrained model")
    parser.add_argument("--model_type", type=str, default='llama3_8b')
    parser.add_argument("--model_name", type=str, default='Meta-Llama-3-8B-Instruct')
    parser.add_argument("--data_path", type=str, required=True, help="Path to the train dataset (JSON)")
    parser.add_argument("--val_path", type=str, default='', help="Path to the val dataset (JSON) if none using k-fold")
    parser.add_argument("--method", type=str, default='icl', help="selecting pruning method")

    parser.add_argument("--dataset", type=str, required=True, help="selecting dataset")
    parser.add_argument("--parallel_batches", type=int, default=5, help="Batch size for parallel inference")
    parser.add_argument("--batch_train", type=int, default=5, help="Batch size for training examples")
    parser.add_argument("--batch_test", type=int, default=8, help="Batch size for validation examples")
    parser.add_argument("--max_token", type=int, default=8192, help="Maximum tokens for input and output combined")
    parser.add_argument("--k_folds", type=int, default=2, help="Number of folds for cross-validation")
    parser.add_argument("--metric", type=str, required=True, help="Metric name for evaluation")
    parser.add_argument("--output_filtered_path", type=str, required=True, help="Path to save filtered training data")
    parser.add_argument("--attn_layer", type=int, default=None)
    parser.add_argument("--memory_output_file", type=str, default="cuda_memory_usage.txt", help="File to save peak CUDA memory usage")
    parser.add_argument("--gpu_index", type=int, default=0, help="Index of the GPU to monitor (default: 0)")
    parser.add_argument("--log_level", type=str, default='INFO', help="Logging level") 
    parser.add_argument("--save_attention_visualizations", type=bool, default=True, help="Save attention visualizations")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Please check your GPU and PyTorch installation.")

    stop_event = threading.Event()
    memory_dict = {}
    monitor_thread = threading.Thread(
        target=monitor_cuda_memory,
        args=(stop_event, memory_dict, args.gpu_index),
        daemon=True,
    )
    monitor_thread.start()

    start_time = time.time()
    run_pruning(args)
    execution_time = time.time() - start_time

    stop_event.set()
    monitor_thread.join()

    # peak_memory_mb = peak_memory_list[0] if peak_memory_list else 0
    memory_dict["peak_memory_allocated"] = torch.cuda.max_memory_allocated(args.gpu_index) / 1024**2
    memory_dict["peak_memory_reserved"] = torch.cuda.max_memory_reserved(args.gpu_index) / 1024**2
    memory_dict["execution_time_seconds"] = execution_time
    memory_output_file = os.path.join(args.output_filtered_path, "cuda_memory_usage.json")
    with open(memory_output_file, "w") as f:
        json.dump(memory_dict, f, ensure_ascii=False, indent=2)
    # with open(memory_output_file, 'w') as f:
    #     f.write(f"Peak CUDA memory allocated (GPU {args.gpu_index}): {memory_dict['memory_allocated']:.2f} MB\n")
    #     f.write(f"Peak CUDA memory reserved (GPU {args.gpu_index}): {memory_dict['memory_reserved']:.2f} MB\n")
    #     f.write(f"Execution time: {execution_time:.2f} seconds\n")

    print(f"Memory info:\n{json.dumps(memory_dict, ensure_ascii=False, indent=2)}")
    print(f"Memory usage and execution time saved to: {memory_output_file}")