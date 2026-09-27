import argparse
import os
import json
import torch
import numpy as np
import re
from tqdm import tqdm
from PIL import Image
from scipy.optimize import linear_sum_assignment
from transformers import AutoModelForCausalLM, AutoProcessor
from keye_vl_utils import process_vision_info

import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error" 

# --- 1. 核心工具函数 ---

def solve_matching(pred_boxes, gt_boxes, iou_threshold=0.5):
    if len(gt_boxes) == 0:
        return 0, len(pred_boxes), 0
    if len(pred_boxes) == 0:
        return 0, 0, len(gt_boxes)

    pred_boxes, gt_boxes = np.array(pred_boxes), np.array(gt_boxes)
    b1, b2 = pred_boxes[:, None, :], gt_boxes[None, :, :]
    inter = np.maximum(0, np.minimum(b1[..., 2:], b2[..., 2:]) - np.maximum(b1[..., :2], b2[..., :2])).prod(-1)
    area1 = (b1[..., 2:] - b1[..., :2]).prod(-1)
    area2 = (b2[..., 2:] - b2[..., :2]).prod(-1)
    iou = inter / (area1 + area2 - inter + 1e-6)

    row_ind, col_ind = linear_sum_assignment(-iou)
    tp = sum(1 for r, c in zip(row_ind, col_ind) if iou[r, c] >= iou_threshold)
    fp = len(pred_boxes) - tp
    fn = len(gt_boxes) - tp
    return tp, fp, fn

def parse_label_studio_bbox(proposal, img_w, img_h):
    return [proposal["x"] * img_w / 100, proposal["y"] * img_h / 100, 
            (proposal["x"] + proposal["width"]) * img_w / 100, (proposal["y"] + proposal["height"]) * img_h / 100]

def parse_keye_output(text, img_w, img_h):
    """提取 Keye 风格的所有坐标块 [[xmin, ymin, xmax, ymax]]"""
    preds = []
    pattern = r"\[\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]\]"
    matches = re.findall(pattern, text)
    for m in matches:
        preds.append([
            int(m[0]) * img_w / 1000, int(m[1]) * img_h / 1000,
            int(m[2]) * img_w / 1000, int(m[3]) * img_h / 1000
        ])
    return preds

# --- 2. 推理包装函数 ---

def run_keye_inference(model, processor, img_path, prompt_text, max_new_tokens=128):
    """封装 Keye-VL 官方推荐的推理流"""
    messages = [{"role": "user", "content": [{"type": "image", "image": img_path}, {"type": "text", "text": prompt_text}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs, mm_processor_kwargs = process_vision_info(messages)
    
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, 
                       padding=True, return_tensors="pt", **mm_processor_kwargs).to("cuda")

    with torch.inference_mode():
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        # 裁剪输入部分
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        response = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0]
    return response

# --- 3. 主评测流程 ---

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_root", type=str, default="test1")
    parser.add_argument("--model_path", type=str, default="/root/autodl-tmp/local_weights/Kwai-Keye/Keye-VL-1_5-8B")
    args = parser.parse_args()

    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.bfloat16, 
                                                 trust_remote_code=True, attn_implementation="flash_attention_2").eval().cuda()
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)

    global_metrics = {"tp": 0, "fp": 0, "fn": 0, "f1_list": []}
    all_dim_results = []
    all_dims = [d for d in os.listdir(args.test_root) if os.path.isdir(os.path.join(args.test_root, d)) and not d.startswith('.')]
    
    for dim in all_dims:
        json_path = os.path.join(args.test_root, f"{dim}_test_labels.json")
        img_dir = os.path.join(args.test_root, dim)
        if not os.path.exists(json_path): continue

        with open(json_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)

        tasks = [item for item in dataset if item.get("positive")]
        dim_tp, dim_fp, dim_fn = 0, 0, 0
        dim_f1s = []
        arb_count = 0

        print(f"\n>>> 评测维度: {dim} | 样本量: {len(tasks)}")

        for item in tqdm(tasks, desc=f"Multi-task Eval {dim}"):
            img_path = os.path.join(img_dir, item["image"])
            if not os.path.exists(img_path): continue
            
            w, h = Image.open(img_path).size
            target = item['positive']

            # --- 任务 A: 计数 ---
            count_prompt = f"Directly count the number of '{target}' in the image. Answer with a single integer."
            count_res = run_keye_inference(model, processor, img_path, count_prompt, max_new_tokens=20)
            
            digit_match = re.search(r'\d+', count_res)
            expected_count = int(digit_match.group()) if digit_match else 1

            # --- 任务 B: 检测 ---
            det_prompt = f"Detect all '{target}' and provide bounding boxes in [[xmin, ymin, xmax, ymax]]."
            det_res = run_keye_inference(model, processor, img_path, det_prompt, max_new_tokens=256)
            preds = parse_keye_output(det_res, w, h)

            # --- 逻辑仲裁 (Arbitration) ---
            if len(preds) != expected_count:
                arb_count += 1
                arb_prompt = (
                    f"Consensus Check: Earlier you counted {expected_count} instances of '{target}', "
                    f"but localized {len(preds)}. Please re-examine the image carefully and provide "
                    f"the final confirmed boxes in [[xmin, ymin, xmax, ymax]]."
                )
                det_res = run_keye_inference(model, processor, img_path, arb_prompt, max_new_tokens=256)
                preds = parse_keye_output(det_res, w, h)

            # 指标计算
            gts = [parse_label_studio_bbox(p, w, h) for p in item.get("proposal", [])]
            tp, fp, fn = solve_matching(preds, gts)
            
            dim_tp += tp; dim_fp += fp; dim_fn += fn
            sample_f1 = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
            dim_f1s.append(sample_f1)
            
            global_metrics["tp"] += tp; global_metrics["fp"] += fp; global_metrics["fn"] += fn
            global_metrics["f1_list"].append(sample_f1)

        d_prec = dim_tp / (dim_tp + dim_fp + 1e-6)
        d_rec = dim_tp / (dim_tp + dim_fn + 1e-6)
        d_f1 = np.mean(dim_f1s) if dim_f1s else 0
        
        print(f"[{dim}] 仲裁率: {arb_count/len(tasks):.2%} | P: {d_prec:.4f} | R: {d_rec:.4f} | F1: {d_f1:.4f}")
        all_dim_results.append({"name": dim, "p": d_prec, "r": d_rec, "f1": d_f1, "tp": dim_tp, "gt": dim_tp + dim_fn})

    print("\n" + "="*85)
    print(f"{'Dimension Name':<20} | {'GT':<6} | {'TP':<6} | {'Precision':<10} | {'Recall':<10} | {'F1':<10}")
    print("-" * 85)
    for res in all_dim_results:
        print(f"{res['name']:<20} | {res['gt']:<6} | {res['tp']:<6} | {res['p']:<10.4f} | {res['r']:<10.4f} | {res['f1']:<10.4f}")
    print("="*85)

if __name__ == "__main__":
    main()
