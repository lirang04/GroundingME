"""
GroundingME Evaluation Script

A standalone script to evaluate vision-language models on the GroundingME benchmark.
Loads data from HuggingFace, calls models via OpenAI-compatible API, and computes evaluation metrics.

Usage:
    python evaluate.py --api-url <api_base_url> --api-key <your_key> --model-name <model_name> [--workers <num>] [--output <output_file>] [--limit <num_samples>]

Examples:
    # Local vLLM server (single worker)
    python evaluate.py --api-url http://localhost:8000/v1 --api-key dummy --model-name Qwen/Qwen3-VL-8B-Thinking
    
    # With concurrent workers (faster evaluation)
    python evaluate.py --api-url http://localhost:8000/v1 --api-key dummy --model-name Qwen/Qwen3-VL-8B-Thinking --workers 16
"""

import argparse
import json
import re
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from datasets import load_dataset
from tqdm import tqdm


PROMPT_TEMPLATE = """All spatial relationships are defined from the viewer's perspective, where 'front' means closer to the viewer and 'back' means farther from the viewer. Please provide the bounding box coordinate of the object the following statement describes:
{description}
Ensure that all details mentioned about the object are accurate. Provide at most one bounding box. If a matching object is found, provide its bounding box as a JSON in the format {{"bbox_2d": [x1, y1, x2, y2]}}. If no matching object is found, output {{"bbox_2d": null}}."""


def parse_bbox(text: str) -> List[float]:
    """Extract bounding box from model response."""
    try:
        match = re.search(r'\{.*"bbox_2d".*\}', text, re.DOTALL)
        if match:
            data = json.loads(match.group(0))
            bbox = data["bbox_2d"]
            if bbox is None:
                return [0, 0, 0, 0]
            if isinstance(bbox, list) and len(bbox) == 4:
                return [float(coord) for coord in bbox]
    except:
        pass
    
    # Fallback: try to extract four numbers
    pattern = r"(-?\d+(?:\.\d+)?)[\s,]+(-?\d+(?:\.\d+)?)[\s,]+(-?\d+(?:\.\d+)?)[\s,]+(-?\d+(?:\.\d+)?)"
    matches = re.findall(pattern, text)
    if matches:
        return [float(coord) for coord in matches[-1]]
    
    return [0, 0, 0, 0]


def compute_iou(box1: List[float], box2: List[float]) -> float:
    """Compute Intersection over Union (IoU) of two bounding boxes."""
    if box1 == [0, 0, 0, 0]:
        return 1 if box2 == [0, 0, 0, 0] else 0
    
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])
    
    intersection = max(0, x_right - x_left) * max(0, y_bottom - y_top)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0


def normalize_bbox(bbox: List[float], width: int, height: int) -> List[float]:
    """Convert normalized or 0-999 range bbox to pixel coordinates."""
    if all(coord <= 1 for coord in bbox):
        return [bbox[0] * width, bbox[1] * height, bbox[2] * width, bbox[3] * height]
    return [bbox[0] / 999 * width, bbox[1] / 999 * height, bbox[2] / 999 * width, bbox[3] / 999 * height]


def call_model(image, prompt: str, api_config: Dict[str, str]) -> str:
    """
    Call vision-language model via OpenAI-compatible API.
    
    Args:
        image: PIL Image object
        prompt: Text prompt
        api_config: Dictionary containing 'base_url', 'api_key', 'model_name'
    
    Returns:
        Model response text
    """
    import base64
    import io
    from openai import OpenAI
    
    # Convert image to base64
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    
    # Initialize client with custom base_url if provided
    client = OpenAI(
        api_key=api_config.get("api_key"),
        base_url=api_config.get("base_url")
    )
    
    response = client.chat.completions.create(
        model=api_config.get("model_name"),
        messages=[{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_str}"}},
                {"type": "text", "text": prompt}
            ]
        }],
        max_completion_tokens=20480,
        temperature=0
    )
    
    return response.choices[0].message.content


def evaluate_sample(sample: Dict[str, Any], api_config: Dict[str, str]) -> Dict[str, Any]:
    """Evaluate a single sample."""
    # Prepare inputs
    image = sample["image"].convert("RGB")
    prompt = PROMPT_TEMPLATE.format(description=sample["description"])
    
    # Get model prediction
    response = call_model(image, prompt, api_config)
    pred_bbox = parse_bbox(response)
    
    # Get ground truth
    gt_bbox = sample["bbox"] if sample["subtask_l1"] != "Rejection" else [0, 0, 0, 0]
    height, width = sample["height"], sample["width"]
    
    # Try different coordinate formats and pick the best one
    pred_candidates = [
        pred_bbox,
        normalize_bbox(pred_bbox, width, height),
    ]
    
    ious = [compute_iou(gt_bbox, pred) for pred in pred_candidates]
    best_idx = ious.index(max(ious))
    best_pred = pred_candidates[best_idx]
    best_iou = ious[best_idx]
    
    # Compute metrics
    acc_50 = float(best_iou >= 0.5)
    acc_75 = float(best_iou >= 0.75)
    acc_90 = float(best_iou >= 0.9)
    
    return {
        "id": sample["id"],
        "subtask_l1": sample["subtask_l1"],
        "subtask_l2": sample["subtask_l2"],
        "iou": best_iou,
        "acc_50": acc_50,
        "acc_75": acc_75,
        "acc_90": acc_90,
        "response": response,
        "pred_bbox": best_pred,
        "gt_bbox": gt_bbox,
    }


def aggregate_results(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """Aggregate evaluation results and compute overall metrics."""
    if not results:
        return {}
    
    # Overall metrics
    metrics = {
        "IoU": sum(r["iou"] for r in results) / len(results),
        "ACC@0.5": sum(r["acc_50"] for r in results) / len(results),
        "ACC@0.75": sum(r["acc_75"] for r in results) / len(results),
        "ACC@0.9": sum(r["acc_90"] for r in results) / len(results),
    }
    
    # Per-category metrics (subtask_l1)
    categories_l1 = {}
    for result in results:
        cat = result["subtask_l1"]
        if cat not in categories_l1:
            categories_l1[cat] = []
        categories_l1[cat].append(result)
    
    for cat, cat_results in categories_l1.items():
        metrics[f"{cat}_ACC@0.5"] = sum(r["acc_50"] for r in cat_results) / len(cat_results)
        metrics[f"{cat}_ACC@0.75"] = sum(r["acc_75"] for r in cat_results) / len(cat_results)
        metrics[f"{cat}_ACC@0.9"] = sum(r["acc_90"] for r in cat_results) / len(cat_results)
    
    # Per-subcategory metrics (subtask_l2)
    categories_l2 = {}
    for result in results:
        if result["subtask_l2"]:
            cat = f"{result['subtask_l1']}_{result['subtask_l2']}"
            if cat not in categories_l2:
                categories_l2[cat] = []
            categories_l2[cat].append(result)
    
    for cat, cat_results in categories_l2.items():
        metrics[f"{cat}_ACC@0.5"] = sum(r["acc_50"] for r in cat_results) / len(cat_results)
    
    return metrics


def evaluate_sample_wrapper(args):
    """Wrapper function for parallel execution."""
    sample, api_config = args
    try:
        return evaluate_sample(sample, api_config), None
    except Exception as e:
        return None, (sample.get('id', 'unknown'), str(e))


def main():
    parser = argparse.ArgumentParser(description="Evaluate models on GroundingME benchmark")
    parser.add_argument("--api-url", type=str, required=True, help="API base URL")
    parser.add_argument("--api-key", type=str, required=True, help="API key for authentication")
    parser.add_argument("--model-name", type=str, required=True, help="Model name")
    parser.add_argument("--dataset", type=str, default="lirang04/GroundingME", help="HuggingFace dataset path")
    parser.add_argument("--split", type=str, default="test", help="Dataset split to evaluate")
    parser.add_argument("--output", type=str, default="results.json", help="Output file for results")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples to evaluate")
    parser.add_argument("--workers", type=int, default=1, help="Number of concurrent workers (default: 1)")
    args = parser.parse_args()
    
    api_config = {
        "base_url": args.api_url,
        "api_key": args.api_key,
        "model_name": args.model_name,
    }
    
    print(f"Loading dataset: {args.dataset} (split: {args.split})")
    dataset = load_dataset(args.dataset, split=args.split)
    
    if args.limit:
        dataset = dataset.select(range(min(args.limit, len(dataset))))
    
    print(f"Evaluating {len(dataset)} samples")
    print(f"  API URL: {args.api_url}")
    print(f"  Model: {args.model_name}")
    print(f"  Workers: {args.workers}")
    
    results = []
    errors = []
    
    if args.workers == 1:
        # Sequential evaluation
        for sample in tqdm(dataset, desc="Evaluating"):
            try:
                result = evaluate_sample(sample, api_config)
                results.append(result)
            except Exception as e:
                error_msg = f"Sample {sample.get('id', 'unknown')}: {e}"
                errors.append(error_msg)
                print(f"\nError: {error_msg}")
    else:
        # Parallel evaluation
        tasks = [(sample, api_config) for sample in dataset]
        
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(evaluate_sample_wrapper, task) for task in tasks]
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="Evaluating"):
                result, error = future.result()
                if result:
                    results.append(result)
                if error:
                    error_msg = f"Sample {error[0]}: {error[1]}"
                    errors.append(error_msg)
                    print(f"\nError: {error_msg}")
    
    # Compute aggregate metrics
    metrics = aggregate_results(results)
    
    # Print results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Total samples: {len(dataset)}")
    print(f"Successful: {len(results)}")
    print(f"Failed: {len(errors)}")
    print("-"*50)
    for metric, value in sorted(metrics.items()):
        print(f"{metric:40s}: {value:.4f}")
    
    # Save detailed results
    output_data = {
        "model": args.model_name,
        "api_url": args.api_url,
        "dataset": args.dataset,
        "split": args.split,
        "workers": args.workers,
        "total_samples": len(dataset),
        "successful_samples": len(results),
        "failed_samples": len(errors),
        "metrics": metrics,
        "detailed_results": results,
        "errors": errors if errors else None,
    }
    
    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nDetailed results saved to: {args.output}")
    if errors:
        print(f"{len(errors)} samples failed - see 'errors' field in output file")


if __name__ == "__main__":
    main()

