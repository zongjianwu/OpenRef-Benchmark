import argparse
import os
import json
import torch
import numpy as np
import math
import re
from tqdm import tqdm
from PIL import Image
from scipy.optimize import linear_sum_assignment
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

# --- 1. 工具函数 ---

def solve_matching(pred_boxes, gt_boxes, iou_threshold=0.5):
    if len(pred_boxes) == 0 or len(gt_boxes) == 0: return 0
    pred_boxes, gt_boxes = np.array(pred_boxes), np.array(gt_boxes)
    b1, b2 = pred_boxes[:, None, :], gt_boxes[None, :, :]
    inter = np.maximum(0, np.minimum(b1[...,2:], b2[...,2:]) - np.maximum(b1[...,:2], b2[...,:2])).prod(-1)
    area1, area2 = (b1[...,2:]-b1[...,:2]).prod(-1), (b2[...,2:]-b2[...,:2]).prod(-1)
    iou = inter / (area1 + area2 - inter + 1e-6)
    row_ind, col_ind = linear_sum_assignment(-iou)
    return sum(1 for r, c in zip(row_ind, col_ind) if iou[r, c] >= iou_threshold)

def parse_label_studio_bbox(proposal):
    W, H = proposal["original_width"], proposal["original_height"]
    return [proposal["x"]*W/100, proposal["y"]*H/100, (proposal["x"]+proposal["width"])*W/100, (proposal["y"]+proposal["height"])*H/100]

def extract_count(text):
    """从文本中提取第一个出现的整数作为计数结果"""
    nums = re.findall(r'\d+', text)
    return int(nums[0]) if nums else 0

def parse_qwen_output(output_text, img_w, img_h):
    preds = []
    try:
        match = re.search(r'\[.*\]', output_text, re.DOTALL)
        if match:
            nums = re.findall(r'\d+', match.group())
            for i in range(0, len(nums) // 4 * 4, 4):
                b = [float(x) for x in nums[i:i+4]]
                preds.append([
                    b[0] * img_w / 1000, 
                    b[1] * img_h / 1000, 
                    b[2] * img_w / 1000, 
                    b[3] * img_h / 1000
                ])
    except Exception as e:
        pass
    return np.array(preds)

def get_model_responses(model, processor, imgs, prompts):
    """通用的批量推理函数"""
    texts = [
        processor.apply_chat_template(
            [{"role": "user", "content": [{"type": "image", "image": i}, {"type": "text", "text": p}]}],
            tokenize=False, add_generation_prompt=True
        ) for i, p in zip(imgs, prompts)
    ]
    
    inputs = processor(text=texts, images=imgs, return_tensors="pt", padding=True).to(model.device)
    
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=256)
        
    generated_ids = generated_ids[:, inputs.input_ids.shape[1]:]
    return processor.batch_decode(generated_ids, skip_special_tokens=True)

# --- 3. 主程序 ---

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", type=str, required=True)
    parser.add_argument("--img_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2-VL-7B-Instruct")
    args = parser.parse_args()

    model = Qwen2VLForConditionalGeneration.from_pretrained(args.model_id, torch_dtype=torch.bfloat16, device_map="auto")
    processor = AutoProcessor.from_pretrained(args.model_id)
    processor.tokenizer.padding_side = "left"

    with open(args.json_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    # 准备任务
    tasks = [item for item in raw_data if item.get("positive")]
    pos_results = {"tp": 0, "fp": 0, "fn": 0, "f1_list": []}
    conflict_count = 0

    for i in tqdm(range(0, len(tasks), args.batch_size), desc="MCC Processing"):
        batch = tasks[i : i + args.batch_size]
        batch_imgs = [Image.open(os.path.join(args.img_dir, t["image"])).convert("RGB") for t in batch]
        
        # --- Stage 1: 并行获取计数和初步检测 ---
        prompt_count = [f"How many '{t['positive']}' are in the image? Answer with an integer." for t in batch]
        prompt_det = [f"Detect all '{t['positive']}'. Return JSON list with 'bbox_2d'." for t in batch]
        
        res_counts = get_model_responses(model, processor, batch_imgs, prompt_count)
        res_dets = get_model_responses(model, processor, batch_imgs, prompt_det)
        
        final_preds_list = []
        
        # --- Stage 2: 冲突检测 ---
        arb_indices = []
        arb_imgs = []
        arb_prompts = []

        for idx, (c_text, d_text) in enumerate(zip(res_counts, res_dets)):
            w, h = batch_imgs[idx].size
            n_count = extract_count(c_text)
            initial_preds = parse_qwen_output(d_text, w, h)
            
            # 记录初始结果
            final_preds_list.append(initial_preds)

            if n_count != len(initial_preds):
                conflict_count += 1
                arb_indices.append(idx)
                arb_imgs.append(batch_imgs[idx])
                arb_prompts.append(
                    f"Inconsistency detected: I counted {n_count} '{batch[idx]['positive']}' but found {len(initial_preds)} boxes. "
                    f"Please re-scan the image carefully and provide the final complete JSON list of all 'bbox_2d'."
                )

        # --- Stage 3: 针对冲突样本进行仲裁 ---
        if arb_imgs:
            res_arbs = get_model_responses(model, processor, arb_imgs, arb_prompts)
            for sub_idx, arb_text in enumerate(res_arbs):
                real_idx = arb_indices[sub_idx]
                w, h = arb_imgs[sub_idx].size
                second_preds = parse_qwen_output(arb_text, w, h)
                
                # 保底策略：只有当二轮仲裁有结果时，才更新一轮结果
                if len(second_preds) > 0:
                    final_preds_list[real_idx] = second_preds

        # --- 指标计算 ---
        for idx, preds in enumerate(final_preds_list):
            gt_boxes = np.array([parse_label_studio_bbox(p) for p in batch[idx]["proposal"]])
            tp = solve_matching(preds, gt_boxes)
            fp, fn = len(preds) - tp, len(gt_boxes) - tp
            
            pos_results["tp"] += tp
            pos_results["fp"] += fp
            pos_results["fn"] += fn
            pos_results["f1_list"].append((2*tp)/(2*tp+fp+fn) if (2*tp+fp+fn)>0 else 0)

    # --- 报告生成 ---
    p = pos_results["tp"] / (pos_results["tp"] + pos_results["fp"] + 1e-6)
    r = pos_results["tp"] / (pos_results["tp"] + pos_results["fn"] + 1e-6)
    f1 = 2 * p * r / (p + r + 1e-6)
    
    print("\n" + "="*60)
    print(f"MCC EVALUATION REPORT")
    print("="*60)
    print(f"Conflict Rate:  {conflict_count/len(tasks):.2%}")
    print(f"Precision:      {p:.4f}")
    print(f"Recall:         {r:.4f}")
    print(f"F1 Score:       {f1:.4f}")
    print(f"Mean F1:        {np.mean(pos_results['f1_list']):.4f}")
    print("="*60)

if __name__ == "__main__":
    main()
