#!/usr/bin/env python3
"""
vLLM Batch Inference for Multiple JSONL Files

This script processes multiple JSONL files with vLLM while preserving the original
file structure. It's optimized for a 600M parameter model on a 40GB A100 GPU.

Usage:
    python vllm_batch_inference.py --model_path /path/to/model --input_dir /path/to/jsonl/files --output_dir /path/to/output

Features:
- Preserves original JSONL file structure
- Efficient batching within and across files
- Checkpointing to resume interrupted jobs
- Progress tracking and estimated time remaining
- Resource monitoring
"""

import os
import json
import time
import glob
import argparse
import logging
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field
import torch
from tqdm import tqdm
import numpy as np
from vllm import LLM, SamplingParams

from prompts import *

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("vllm_inference.log")
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class InferenceItem:
    """Represents a single inference item with tracking information."""
    prompt: str
    metadata: Dict[str, Any]
    source_file: str
    line_idx: int
    result: Optional[str] = None
    tokens_generated: int = 0
    inference_time: float = 0.0


@dataclass
class BatchInferenceTracker:
    """Tracks batches and progress across multiple files."""
    total_items: int = 0
    processed_items: int = 0
    current_batch: List[InferenceItem] = field(default_factory=list)
    results: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    checkpoint_path: str = "inference_checkpoint.json"
    start_time: float = field(default_factory=time.time)
    
    def add_item(self, item: InferenceItem) -> None:
        """Add an item to the current batch."""
        self.current_batch.append(item)
    
    def update_progress(self, completed_items: List[InferenceItem]) -> None:
        """Update progress with completed items."""
        self.processed_items += len(completed_items)
        
        # Store results by source file
        for item in completed_items:
            if item.source_file not in self.results:
                self.results[item.source_file] = []
            
            result_dict = {
                **item.metadata,
                "prompt": item.prompt,
                "generated_text": item.result,
                "tokens_generated": item.tokens_generated,
                "inference_time_ms": item.inference_time * 1000
            }
            
            # Insert at the original line index position
            while len(self.results[item.source_file]) <= item.line_idx:
                self.results[item.source_file].append(None)
            self.results[item.source_file][item.line_idx] = result_dict
    
    def get_progress_stats(self) -> Dict[str, Any]:
        """Get progress statistics."""
        elapsed = time.time() - self.start_time
        items_per_sec = self.processed_items / elapsed if elapsed > 0 else 0
        remaining = (self.total_items - self.processed_items) / items_per_sec if items_per_sec > 0 else 0
        
        return {
            "processed": self.processed_items,
            "total": self.total_items,
            "percent": (self.processed_items / self.total_items * 100) if self.total_items > 0 else 0,
            "elapsed_sec": elapsed,
            "items_per_sec": items_per_sec,
            "est_remaining_sec": remaining
        }
    
    def save_checkpoint(self) -> None:
        """Save progress checkpoint."""
        # Convert results to a serializable format
        serializable_results = {}
        for file_name, items in self.results.items():
            serializable_results[file_name] = [item for item in items if item is not None]
        
        checkpoint_data = {
            "processed_items": self.processed_items,
            "total_items": self.total_items,
            "results": serializable_results,
            "timestamp": time.time()
        }
        
        with open(self.checkpoint_path, 'w') as f:
            json.dump(checkpoint_data, f)
        
        logger.info(f"Checkpoint saved: {self.processed_items}/{self.total_items} items processed")
    
    def load_checkpoint(self) -> bool:
        """Load progress checkpoint if exists."""
        if not os.path.exists(self.checkpoint_path):
            return False
        
        try:
            with open(self.checkpoint_path, 'r') as f:
                checkpoint_data = json.load(f)
            
            self.processed_items = checkpoint_data["processed_items"]
            self.total_items = checkpoint_data["total_items"]
            
            # Load results
            for file_name, items in checkpoint_data["results"].items():
                self.results[file_name] = []
                for i, item in enumerate(items):
                    while len(self.results[file_name]) <= i:
                        self.results[file_name].append(None)
                    self.results[file_name][i] = item
            
            logger.info(f"Checkpoint loaded: {self.processed_items}/{self.total_items} items already processed")
            return True
            
        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}")
            return False


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run vLLM inference on multiple JSONL files")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing JSONL files")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory for output files")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for inference")
    parser.add_argument("--prompt_field", type=str, default="prompt", help="Field name containing the prompt")
    parser.add_argument("--max_tokens", type=int, default=2048, help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--checkpoint_interval", type=int, default=50, help="Save checkpoint every N batches")
    parser.add_argument("--file_pattern", type=str, default="*.jsonl", help="JSONL file pattern to match")
    parser.add_argument("--skip_processed", action="store_true", help="Skip already processed files")
    parser.add_argument("--num_rounds", type=int, default=1, help="Number of inference rounds to run")
    parser.add_argument("--run_id", type=str, default="", help="Optional identifier for this run")
    parser.add_argument("--language", type=str, default="zh", help="Experiment language")
    parser.add_argument("--cot", type=bool, default=False, help="Prompt chain-of-thought, not used with reasoning models")

    return parser.parse_args()


def load_jsonl_files(input_dir: str, file_pattern: str, prompt_field: str, lang: str, cot: bool) -> List[Tuple[str, List[InferenceItem]]]:
    """Load all JSONL files and prepare inference items."""
    file_paths = glob.glob(os.path.join(input_dir, file_pattern))
    all_files_data = []
    
    for file_path in sorted(file_paths):
        file_name = os.path.basename(file_path)
        items = []
        
        with open(file_path, 'r') as f:
            for i, line in enumerate(f):
                try:
                    d = json.loads(line.strip())
                    if isinstance(d['选项C'], str):
                        maps, prompt = format_prompt_4(d, lang)
                    else:
                        maps, prompt = format_prompt_2(d, lang)
                
                    system_prompt = ""
                    if lang == "zh":
                        if cot == False:
                            system_prompt = SystemEvaluatePrompt_zh
                        else:
                            system_prompt = SystemEvaluatePrompt_zh_cot
                    else:
                        if cot == False:
                            system_prompt = SystemEvaluatePrompt_en
                        else:
                            system_prompt = SystemEvaluatePrompt_en_cot

                    items.append(InferenceItem(
                        prompt = [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": prompt}
                        ],
                        metadata={
                          'idx': i,
                          'answer': d['答案\nANSWER'],
                          'map': maps,
                          'data': d
                        },
                        source_file=file_name,
                        line_idx=i
                    ))
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON at line {i + 1} in {file_path}")
                    continue
        
        all_files_data.append((file_name, items))
    
    return all_files_data


def run_inference(model: LLM, batch: List[InferenceItem], sampling_params: SamplingParams) -> List[InferenceItem]:
    """Run inference on a batch of items."""
    # Convert chat prompts to Qwen string format
    prompts = [format_prompt_for_qwen(item.prompt) for item in batch]

    # Measure inference time
    start_time = time.time()
    outputs = model.generate(prompts, sampling_params)
    inference_time = time.time() - start_time
    
    # Process outputs
    for i, output in enumerate(outputs):
        batch[i].result = output.outputs[0].text
        batch[i].tokens_generated = len(output.outputs[0].token_ids)
        batch[i].inference_time = inference_time / len(batch)  # Approximate per-item time
    
    return batch


def save_results(tracker: BatchInferenceTracker, output_dir: str, round_idx: int = 0, run_id: str = "") -> None:
    """Save results to output files, preserving original structure."""
    
    # Create round-specific directory if multiple rounds
    round_suffix = f"_round{round_idx}" if round_idx > 0 else ""
    run_prefix = f"{run_id}_" if run_id else ""
    
    for file_name, results in tracker.results.items():
        # Create output filename with round index
        base_name, ext = os.path.splitext(file_name)
        output_filename = f"{run_prefix}{base_name}{round_suffix}{ext}"
        output_path = os.path.join(output_dir, output_filename)
        
        with open(output_path, 'w') as f:
            for item in results:
                if item is not None:
                    # Add round information to the output
                    item_with_round = {**item, "inference_round": round_idx}
                    f.write(json.dumps(item_with_round) + '\n')
        
        logger.info(f"Saved results to {output_path}")


def format_time(seconds: float) -> str:
    """Format seconds into a human-readable string."""
    hours, remainder = divmod(int(seconds), 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

def format_prompt_for_qwen(chat_prompt):
    """Convert a chat format prompt to a string format for Qwen models."""
    result = ""
    
    for message in chat_prompt:
        if message["role"] == "system":
            result += f"<|im_start|>system\n{message['content']}<|im_end|>\n"
        elif message["role"] == "user":
            result += f"<|im_start|>user\n{message['content']}<|im_end|>\n"
        elif message["role"] == "assistant":
            result += f"<|im_start|>assistant\n{message['content']}<|im_end|>\n"
    
    # Add the assistant prefix for the response
    result += "<|im_start|>assistant\n"
    
    return result

def run_inference_round(
    model: LLM, 
    all_files_data: List[Tuple[str, List[InferenceItem]]], 
    args: argparse.Namespace, 
    round_idx: int = 0
) -> None:
    """Run a complete inference round."""
    # Initialize tracking
    checkpoint_path = os.path.join(
        args.output_dir, 
        f"inference_checkpoint_round{round_idx}.json"
    )
    tracker = BatchInferenceTracker(checkpoint_path=checkpoint_path)
    
    # Load checkpoint if exists
    checkpoint_loaded = tracker.load_checkpoint()
    
    # Prepare sampling parameters
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=0.95,
        max_tokens=args.max_tokens
    )
    
    # Calculate total number of items
    total_items = sum(len(items) for _, items in all_files_data)
    tracker.total_items = total_items
    
    logger.info(f"Round {round_idx}: Found {len(all_files_data)} files with {total_items} total items")
    
    # Process files and batches
    current_batch = []
    batch_count = 0
    
    # Process all files
    with tqdm(total=total_items, initial=tracker.processed_items) as pbar:
        for file_name, items in all_files_data:
            # Skip already processed files if needed
            if args.skip_processed and file_name in tracker.results and len(tracker.results[file_name]) > 0:
                logger.info(f"Round {round_idx}: Skipping already processed file: {file_name}")
                pbar.update(len(items))
                continue
                
            # Process items in batches
            for item in items:
                # Skip already processed items (from checkpoint)
                if file_name in tracker.results and len(tracker.results[file_name]) > item.line_idx and tracker.results[file_name][item.line_idx] is not None:
                    pbar.update(1)
                    continue
                    
                current_batch.append(item)
                
                # Process batch when full
                if len(current_batch) >= args.batch_size:
                    processed_items = run_inference(model, current_batch, sampling_params)
                    tracker.update_progress(processed_items)
                    pbar.update(len(processed_items))
                    
                    # Display progress info
                    stats = tracker.get_progress_stats()
                    pbar.set_description(
                        f"Round {round_idx}: Speed: {stats['items_per_sec']:.2f} items/s | "
                        f"ETA: {format_time(stats['est_remaining_sec'])}"
                    )
                    
                    # Save checkpoint periodically
                    batch_count += 1
                    if batch_count % args.checkpoint_interval == 0:
                        save_results(tracker, args.output_dir, round_idx, args.run_id)
                        tracker.save_checkpoint()
                    
                    current_batch = []
        
        # Process final batch if any
        if current_batch:
            processed_items = run_inference(model, current_batch, sampling_params)
            tracker.update_progress(processed_items)
            pbar.update(len(processed_items))
    
    # Save final results
    logger.info(f"Round {round_idx}: Inference completed, saving final results")
    save_results(tracker, args.output_dir, round_idx, args.run_id)
    
    # Print completion statistics
    stats = tracker.get_progress_stats()
    logger.info(f"Round {round_idx}: Processed {tracker.processed_items}/{tracker.total_items} items "
                f"({stats['percent']:.2f}%) in {format_time(stats['elapsed_sec'])}")
    logger.info(f"Round {round_idx}: Average speed: {stats['items_per_sec']:.2f} items/sec")
    
    # Clean up checkpoint
    if os.path.exists(tracker.checkpoint_path):
        os.remove(tracker.checkpoint_path)
        logger.info(f"Round {round_idx}: Checkpoint file removed")


def main():
    """Main execution function."""
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize model
    logger.info(f"Initializing vLLM with model: {args.model_path}")
    model = LLM(
        model=args.model_path,
        tensor_parallel_size=2,  # number of GPUs
        gpu_memory_utilization=0.9,
        max_num_batched_tokens=8192
    )
    
    # Load all JSONL files
    logger.info(f"Loading JSONL files from {args.input_dir}")
    all_files_data = load_jsonl_files(args.input_dir, args.file_pattern, args.prompt_field, args.language, args.cot)
    
    # Run inference for specified number of rounds
    for round_idx in range(args.num_rounds):
        logger.info(f"Starting inference round {round_idx} of {args.num_rounds}")
        run_inference_round(model, all_files_data, args, round_idx)


if __name__ == "__main__":
    main()

