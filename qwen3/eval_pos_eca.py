import argparse
import os
import json
import torch
import numpy as np
import re
from tqdm import tqdm
from PIL import Image
from scipy.optimize import linear_sum_assignment
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor


def solve_matching(pred_boxes, gt_boxes, iou_threshold=0.5):
    """ 计算 TP (True Positives) 基于匈牙利匹配 """
    if len(pred_boxes) == 0 or len(gt_boxes) == 0: return 0
    pred_boxes, gt_boxes = np.array(pred_boxes), np.array(gt_boxes)
    b1, b2 = pred_boxes[:, None, :], gt_boxes[None, :, :]
    inter = np.maximum(0, np.minimum(b1[...,2:], b2[...,2:]) - np.maximum(b1[...,:2], b2[...,:2])).prod(-1)
    area1, area2 = (b1[...,2:]-b1[...,:2]).prod(-1), (b2[...,2:]-b2[...,:2]).prod(-1)
    iou = inter / (area1 + area2 - inter + 1e-6)
    row_ind, col_ind = linear_sum_assignment(-iou)
    return sum(1 for r, c in zip(row_ind, col_ind) if iou[r, c] >= iou_threshold)

def parse_label_studio_bbox(proposal):
    """ 将 Label Studio 归一化格式转为像素坐标 """
    W, H = proposal["original_width"], proposal["original_height"]
    return [proposal["x"]*W/100, proposal["y"]*H/100, (proposal["x"]+proposal["width"])*W/100, (proposal["y"]+proposal["height"])*H/100]

def parse_qwen_output(output_text, img_w, img_h):
    text_lower = output_text.lower()
    negative_words = ["none", "no object", "not found", "no ", "isn't", "unable", "not detect"]
    is_rejected = any(word in text_lower for word in negative_words)
    preds = []
    try:
        json_match = re.search(r'\[\s*\{.*\}\s*\]', output_text.replace("'", '"'), re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
            for item in data:
                b = item.get("bbox_2d", [])
                if len(b) == 4:
                    preds.append([b[0]*img_w/1000, b[1]*img_h/1000, b[2]*img_w/1000, b[3]*img_h/1000])
            if len(preds) > 0: is_rejected = False
    except:
        pass
    return np.array(preds), is_rejected


def run_batch_inference(model, processor, batch_imgs, batch_prompts, max_tokens=256):
    processor.tokenizer.padding_side = "left"
    texts = [processor.apply_chat_template(
        [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": p}]}],
        tokenize=False, add_generation_prompt=True
    ) for img, p in zip(batch_imgs, batch_prompts)]
    
    inputs = processor(text=texts, images=batch_imgs, return_tensors="pt", padding=True).to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=max_tokens, do_sample=False,
            pad_token_id=processor.tokenizer.pad_token_id
        )
    
    generated_ids = outputs[:, inputs.input_ids.shape[1]:]
    return processor.batch_decode(generated_ids, skip_special_tokens=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", type=str, default="/root/autodl-tmp/test/single_test_labels.json")
    parser.add_argument("--img_dir", type=str, default="/root/autodl-tmp/test/single")
    parser.add_argument("--batch_size", type=int, default=8) 
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen3-VL-2B-Instruct") # 或 Qwen2-VL
    args = parser.parse_args()

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_id, torch_dtype=torch.bfloat16, device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(args.model_id)
    
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
    neg_results = {"count": 0, "strict_sum": 0}

    for i in tqdm(range(0, len(task_queue), args.batch_size), desc="Aligning REC to Count"):
        batch_tasks = task_queue[i : i + args.batch_size]
        batch_imgs = [Image.open(t["img"]).convert("RGB") for t in batch_tasks]
        
        # Step 1: 计数推理 (Counting)
        count_prompts = [f"Directly output the number of '{t['query']}' in the image. Answer with a single integer." for t in batch_tasks]
        raw_counts_text = run_batch_inference(model, processor, batch_imgs, count_prompts, max_tokens=20)
        
        # Step 2: 初始检测推理 (REC)
        det_prompts = [f"Locate all '{t['query']}'. Output a JSON list of 'bbox_2d' [xmin, ymin, xmax, ymax]." for t in batch_tasks]
        raw_dets_text = run_batch_inference(model, processor, batch_imgs, det_prompts, max_tokens=300)
        
        final_preds_batch = []
        final_is_rej_batch = []
        
        align_indices = []
        align_imgs = []
        align_prompts = []
        
        for idx, (c_text, d_text) in enumerate(zip(raw_counts_text, raw_dets_text)):
            w, h = batch_imgs[idx].size
            # 提取计数结果
            c_match = re.findall(r'\d+', c_text)
            exp_count = int(c_match[0]) if c_match else 0
            
            initial_preds, initial_rej = parse_qwen_output(d_text, w, h)
            
            # --- 核心对齐逻辑 ---
            # 如果 REC 的数量与 Counting 结果不一致，则发起对齐
            if len(initial_preds) != exp_count:
                align_indices.append(idx)
                align_imgs.append(batch_imgs[idx])
                
                if exp_count == 0:
                    p = f"Visual analysis confirms there are 0 '{batch_tasks[idx]['query']}'. Please output an empty list [] for bbox_2d."
                else:
                    p = (f"Observation: There are exactly {exp_count} instances of '{batch_tasks[idx]['query']}' here.\n"
                         f"Please provide the bounding boxes for these {exp_count} instances in JSON format: [{{'bbox_2d': [xmin, ymin, xmax, ymax], 'label': '...'}}, ...]")
                align_prompts.append(p)
            
            final_preds_batch.append(initial_preds)
            final_is_rej_batch.append(initial_rej)

        if align_indices:
            align_results = run_batch_inference(model, processor, align_imgs, align_prompts, max_tokens=400)
            for sub_idx, res_text in enumerate(align_results):
                real_idx = align_indices[sub_idx]
                w, h = batch_imgs[real_idx].size
                
                aligned_preds, aligned_rej = parse_qwen_output(res_text, w, h)
                final_preds_batch[real_idx] = aligned_preds
                final_is_rej_batch[real_idx] = aligned_rej

        for idx, task in enumerate(batch_tasks):
            preds = final_preds_batch[idx]
            if task["type"] == "pos":
                gt_boxes = np.array([parse_label_studio_bbox(p) for p in task["gt"]])
                tp = solve_matching(preds, gt_boxes)
                fp, fn = len(preds) - tp, len(gt_boxes) - tp
                pos_results["tp"] += tp
                pos_results["fp"] += fp
                pos_results["fn"] += fn
                pos_results["f1_list"].append((2*tp)/(2*tp+fp+fn) if (2*tp+fp+fn)>0 else 0)
            else:
                neg_results["count"] += 1
                if len(preds) == 0:
                    neg_results["strict_sum"] += 1

    print("\n" + "="*60)
    print(f"REPORT: ALIGNED REC-COUNTING EVAL ({args.model_id})")
    print("="*60)
    if pos_results["f1_list"]:
        p = pos_results["tp"] / (pos_results["tp"] + pos_results["fp"] + 1e-6)
        r = pos_results["tp"] / (pos_results["tp"] + pos_results["fn"] + 1e-6)
        print(f"Positive Sample -> Precision: {p:.4f} | Recall: {r:.4f} | Mean F1: {np.mean(pos_results['f1_list']):.4f}")
    if neg_results["count"] > 0:
        print(f"Negative Sample -> NSR (Negative Success Rate): {neg_results['strict_sum'] / neg_results['count']:.4f}")
    print("="*60)

if __name__ == "__main__":
    main()