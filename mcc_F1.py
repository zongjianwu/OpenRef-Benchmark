import argparse
import os
import json
import torch
import numpy as np
import re
from tqdm import tqdm
from PIL import Image
from scipy.optimize import linear_sum_assignment
from transformers import AutoModelForImageTextToText, AutoProcessor

# --- 1. 工具函数 ---

def solve_matching(pred_boxes, gt_boxes, iou_threshold=0.5):
    pred_boxes = np.array(pred_boxes, dtype=np.float64)
    gt_boxes = np.array(gt_boxes, dtype=np.float64)
    if pred_boxes.ndim != 2 or pred_boxes.shape[-1] != 4:
        if pred_boxes.size == 0: return 0
        pred_boxes = pred_boxes.reshape(-1, 4) if pred_boxes.size % 4 == 0 else np.array([]).reshape(0, 4)
    if gt_boxes.ndim != 2 or gt_boxes.shape[-1] != 4:
        if gt_boxes.size == 0: return 0
        gt_boxes = gt_boxes.reshape(-1, 4) if gt_boxes.size % 4 == 0 else np.array([]).reshape(0, 4)
    if len(pred_boxes) == 0 or len(gt_boxes) == 0: return 0
    b1, b2 = pred_boxes[:, None, :], gt_boxes[None, :, :]
    inter_min = np.maximum(b1[..., :2], b2[..., :2])
    inter_max = np.minimum(b1[..., 2:], b2[..., 2:])
    inter_wh = np.maximum(0, inter_max - inter_min)
    inter_area = inter_wh[..., 0] * inter_wh[..., 1]
    area1 = (b1[..., 2] - b1[..., 0]) * (b1[..., 3] - b1[..., 1])
    area2 = (b2[..., 2] - b2[..., 0]) * (b2[..., 3] - b2[..., 1])
    union_area = area1 + area2 - inter_area
    iou = inter_area / (union_area + 1e-6)
    row_ind, col_ind = linear_sum_assignment(-iou)
    tp = sum(1 for r, c in zip(row_ind, col_ind) if iou[r, c] >= iou_threshold)
    return tp

# [保留你原有的 parse_qwen_json_output 函数不变]
def parse_qwen_json_output(text):
    preds = []
    match = re.search(r'\[\s*\{.*\}\s*\]', text, re.DOTALL)
    if match:
        try:
            clean_text = match.group()
            data = json.loads(clean_text)
            for item in data:
                if "bbox_2d" in item:
                    box = item["bbox_2d"]
                    if isinstance(box, list) and len(box) == 4:
                        preds.append([float(x) for x in box])
        except Exception:
            raw_boxes = re.findall(r'\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]', text)
            for rb in raw_boxes:
                preds.append([float(x) for x in rb])
    return np.array(preds) if preds else np.zeros((0, 4))

# [新增] 专门解析计数的函数
def parse_count_output(text):
    nums = re.findall(r'\d+', text)
    return int(nums[0]) if nums else 0

def parse_label_studio_bbox(proposal):
    W, H = proposal["original_width"], proposal["original_height"]
    return [proposal["x"] * W / 100, proposal["y"] * H / 100, 
            (proposal["x"] + proposal["width"]) * W / 100, (proposal["y"] + proposal["height"]) * H / 100]

# --- 2. 核心执行逻辑 ---

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_root", type=str, required=True)
    parser.add_argument("--model_id", type=str, default="/root/autodl-tmp/local_weights/qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--batch_size", type=int, default=1)
    args = parser.parse_args()

    model = AutoModelForImageTextToText.from_pretrained(args.model_id, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)

    all_dims = [d for d in os.listdir(args.test_root) if os.path.isdir(os.path.join(args.test_root, d)) and d != "none"]
    dimension_results = {}

    for dim in all_dims:
        json_path = os.path.join(args.test_root, f"{dim}_test_labels.json")
        img_dir = os.path.join(args.test_root, dim)
        if not os.path.exists(json_path): continue

        with open(json_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)

        tasks = [item for item in dataset if item.get("positive")]
        dim_tp, dim_fp, dim_fn = 0, 0, 0
        total_conflicts = 0 # 记录冲突次数

        print(f"\n>>> Evaluating Dimension: {dim} ({len(tasks)} samples) with MCC")
        
        for i in tqdm(range(0, len(tasks), args.batch_size)):
            batch = tasks[i : i + args.batch_size]
            batch_imgs = [Image.open(os.path.join(img_dir, item["image"])).convert("RGB") for item in batch]
            
            # --- Stage 1 & 2: 同时获取计数和检测结果 ---
            batch_count_prompts = []
            batch_det_prompts = []
            
            for idx, item in enumerate(batch):
                # 计数 Prompt
                cnt_prompt = processor.apply_chat_template([{"role": "user", "content": [
                    {"type": "image", "image": batch_imgs[idx]},
                    {"type": "text", "text": f"How many '{item['positive']}' are in the image? Answer with an integer only."}
                ]}], tokenize=False, add_generation_prompt=True)
                batch_count_prompts.append(cnt_prompt)
                
                # 检测 Prompt (完全保持你原来的 Prompt 不变)
                det_prompt = processor.apply_chat_template([{"role": "user", "content": [
                    {"type": "image", "image": batch_imgs[idx]},
                    {"type": "text", "text": f"Detect the object: {item['positive']}. Return JSON list with 'bbox_2d' [xmin, ymin, xmax, ymax]"}
                ]}], tokenize=False, add_generation_prompt=True)
                batch_det_prompts.append(det_prompt)

            # 推理计数
            inputs_cnt = processor(text=batch_count_prompts, images=batch_imgs, return_tensors="pt", padding=True).to(model.device)
            with torch.no_grad():
                ids_cnt = model.generate(**inputs_cnt, max_new_tokens=50)
                res_cnt = processor.batch_decode(ids_cnt[:, inputs_cnt.input_ids.shape[1]:], skip_special_tokens=True)
            
            # 推理检测
            inputs_det = processor(text=batch_det_prompts, images=batch_imgs, return_tensors="pt", padding=True).to(model.device)
            with torch.no_grad():
                ids_det = model.generate(**inputs_det, max_new_tokens=512)
                res_det = processor.batch_decode(ids_det[:, inputs_det.input_ids.shape[1]:], skip_special_tokens=True)

            # --- Stage 3: 冲突判定与仲裁 ---
            final_batch_preds = []
            # --- 改进的仲裁逻辑 ---
            for idx, (cnt_text, det_text) in enumerate(zip(res_cnt, res_det)):
                predicted_count = parse_count_output(cnt_text)
                initial_preds = parse_qwen_json_output(det_text)
                obj_name = batch[idx]['positive']
                len_initial = len(initial_preds)
                
                if predicted_count != len_initial:
                    total_conflicts += 1
                    
                    # 构造更严谨的仲裁 Prompt
                    arb_prompt = processor.apply_chat_template([{"role": "user", "content": [
                        {"type": "image", "image": batch_imgs[idx]},
                        {"type": "text", "text": (
                            f"Validation Step: I previously counted {predicted_count} '{obj_name}' "
                            f"but found only {len_initial} boxes. One of these is likely wrong. "
                            f"Please re-scan the image carefully. If there are actually {predicted_count} "
                            f"objects, find all their coordinates. Return the final JSON list of 'bbox_2d'."
                        )}
                    ]}], tokenize=False, add_generation_prompt=True)
                    
                    inputs_arb = processor(text=[arb_prompt], images=[batch_imgs[idx]], return_tensors="pt").to(model.device)
                    with torch.no_grad():
                        ids_arb = model.generate(**inputs_arb, max_new_tokens=512, do_sample=False) 
                        res_arb = processor.batch_decode(ids_arb[:, inputs_arb.input_ids.shape[1]:], skip_special_tokens=True)
                    
                    arb_preds = parse_qwen_json_output(res_arb[0])
                    
                    if len(arb_preds) >= len_initial:
                        final_batch_preds.append(arb_preds)
                    else:
                        final_batch_preds.append(initial_preds)
                else:
                    final_batch_preds.append(initial_preds)

            # 指标累计
            for idx, preds in enumerate(final_batch_preds):
                gts = np.array([parse_label_studio_bbox(p) for p in batch[idx]["proposal"]])
                tp = solve_matching(preds, gts)
                dim_tp += tp
                dim_fp += (len(preds) - tp)
                dim_fn += (len(gts) - tp)

        dimension_results[dim] = {"tp": dim_tp, "fp": dim_fp, "fn": dim_fn, "conflicts": total_conflicts}

    # --- 3. 输出汇总报告 ---
    print("\n" + "="*95)
    print(f"{'Dimension':<20} | {'GT':<6} | {'Conflict':<8} | {'Prec.':<10} | {'Recall':<10} | {'F1-Score'}")
    print("-" * 95)
    
    all_tp, all_fp, all_fn = 0, 0, 0
    for dim, res in dimension_results.items():
        tp, fp, fn = res["tp"], res["fp"], res["fn"]
        gt_count = tp + fn
        p = tp / (tp + fp + 1e-6)
        r = tp / (tp + fn + 1e-6)
        f1 = 2 * p * r / (p + r + 1e-6)
        
        print(f"{dim:<20} | {gt_count:<6} | {res['conflicts']:<8} | {p:9.2%} | {r:9.2%} | {f1:.4f}")
        
        all_tp += tp
        all_fp += fp
        all_fn += fn

    total_p = all_tp / (all_tp + all_fp + 1e-6)
    total_r = all_tp / (all_tp + all_fn + 1e-6)
    total_f1 = 2 * total_p * total_r / (total_p + total_r + 1e-6)

    print("-" * 95)
    print(f"{'OVERALL (Micro)':<20} | {all_tp + all_fn:<6} | {'-':<8} | {total_p:9.2%} | {total_r:9.2%} | {total_f1:.4f}")
    print("="*95)

if __name__ == "__main__":
    main()
