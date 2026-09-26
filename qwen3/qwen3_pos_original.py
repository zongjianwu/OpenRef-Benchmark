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
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

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

def parse_qwen_output(output_text, img_w, img_h):
    text_lower = output_text.lower()
    negative_words = ["none", "no object", "not found", "no ", "isn't", "unable", "not detect", "not present"]
    is_rejected = any(word in text_lower for word in negative_words)
    preds = []
    try:
        json_match = re.search(r'\[\s*\{.*\}\s*\]', output_text, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
            for item in data:
                b = item.get("bbox_2d", [])
                if len(b) == 4:
                    preds.append([b[0]*img_w/1000, b[1]*img_h/1000, b[2]*img_w/1000, b[3]*img_h/1000])
            if len(preds) > 0: is_rejected = False
    except: pass
    return np.array(preds), is_rejected

# --- 2. 主流程 ---

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", type=str, default="/root/autodl-tmp/motivation/openref/occularsion.json")
    parser.add_argument("--img_dir", type=str, default="/root/autodl-tmp/motivation/openref/occularsion")
    parser.add_argument("--batch_size", type=int, default=4) 
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen3-VL-8B-Instruct")
    args = parser.parse_args()

    # 加载模型
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_id, 
        torch_dtype=torch.bfloat16, 
        device_map="auto"
    )
    
    # --- 核心修改点 1: 初始化 Processor 并强制左填充 ---
    processor = AutoProcessor.from_pretrained(args.model_id)
    processor.tokenizer.padding_side = "left"  # 必须设为 left
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    # ----------------------------------------------

    with open(args.json_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    task_queue = []
    for item in raw_data:
        img_path = os.path.join(args.img_dir, item["image"])
        if not os.path.exists(img_path): continue
        if item.get("positive"):
            task_queue.append({"type": "pos", "img": img_path, "query": item["positive"], "gt": item["proposal"]})
        if item.get("negative"):
            task_queue.append({"type": "neg", "img": img_path, "query": item["negative"], "gt": None})

    pos_results = {"tp": 0, "fp": 0, "fn": 0, "f1_list": []}
    neg_results = {"strict_sum": 0, "weighted_sum": 0, "count": 0}

    # Batch 推理
    for i in tqdm(range(0, len(task_queue), args.batch_size), desc="Batch Inferencing"):
        batch_tasks = task_queue[i : i + args.batch_size]
        
        batch_imgs = []
        batch_texts = []
        
        for t in batch_tasks:
            img = Image.open(t["img"]).convert("RGB")
            batch_imgs.append(img)
            prompt = f"Detect '{t['query']}'. Return result in JSON list with 'bbox_2d' and 'label'."
            # 使用官方 chat 模板构建
            text = processor.apply_chat_template(
                [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": prompt}]}], 
                tokenize=False, 
                add_generation_prompt=True
            )
            batch_texts.append(text)

        # --- 核心修改点 2: 确保在处理输入时保持左填充 ---
        processor.tokenizer.padding_side = "left" 
        inputs = processor(
            text=batch_texts, 
            images=batch_imgs, 
            return_tensors="pt", 
            padding=True
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=128, 
                return_dict_in_generate=True, 
                output_scores=True,
                pad_token_id=processor.tokenizer.pad_token_id # 显式传递 pad_token_id
            )
            
        generated_ids = outputs.sequences[:, inputs.input_ids.shape[1]:]
        output_responses = processor.batch_decode(generated_ids, skip_special_tokens=True)
        
        # 计算置信度
        scores = model.compute_transition_scores(outputs.sequences, outputs.scores, normalize_logits=True)
        batch_confs = torch.exp(scores).mean(dim=1).cpu().numpy()

        for idx, task in enumerate(batch_tasks):
            w, h = batch_imgs[idx].size
            preds, is_rej = parse_qwen_output(output_responses[idx], w, h)
            
            if task["type"] == "pos":
                gt_boxes = np.array([parse_label_studio_bbox(p) for p in task["gt"]])
                tp = solve_matching(preds, gt_boxes)
                fp, fn = len(preds) - tp, len(gt_boxes) - tp
                pos_results["tp"] += tp; pos_results["fp"] += fp; pos_results["fn"] += fn
                pos_results["f1_list"].append((2*tp)/(2*tp+fp+fn) if (2*tp+fp+fn)>0 else 0)
            else:
                neg_results["count"] += 1
                if is_rej and len(preds) == 0:
                    neg_results["strict_sum"] += 1
                    neg_results["weighted_sum"] += 1.0
                else:
                    neg_results["weighted_sum"] += math.exp(-1.0 * (len(preds) + batch_confs[idx]))

    print("\n" + "="*60)
    print(f"REPORT (Batch Size: {args.batch_size})")
    print("="*60)
    if pos_results["f1_list"]:
        p = pos_results["tp"] / (pos_results["tp"] + pos_results["fp"] + 1e-6)
        r = pos_results["tp"] / (pos_results["tp"] + pos_results["fn"] + 1e-6)
        print(f"REC Precision: {p:.4f} | Recall: {r:.4f} | Mean F1: {np.mean(pos_results['f1_list']):.4f}")
    print("="*60)

if __name__ == "__main__":
    main()
