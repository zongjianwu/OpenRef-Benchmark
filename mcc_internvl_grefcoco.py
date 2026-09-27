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
import transformers

transformers.utils.logging.set_verbosity_error()

def solve_matching(pred_boxes, gt_boxes, iou_threshold=0.5):
    if len(pred_boxes) == 0 or len(gt_boxes) == 0: return 0
    pred_boxes, gt_boxes = np.array(pred_boxes, dtype=np.float64), np.array(gt_boxes, dtype=np.float64)
    if pred_boxes.ndim == 1: pred_boxes = pred_boxes.reshape(-1, 4)
    if gt_boxes.ndim == 1: gt_boxes = gt_boxes.reshape(-1, 4)

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
    return sum(1 for r, c in zip(row_ind, col_ind) if iou[r, c] >= iou_threshold)

def preprocess_to_32x(image, div=32):
    w, h = image.size
    new_w = int(np.ceil(w / div) * div)
    new_h = int(np.ceil(h / div) * div)
    padded_img = Image.new('RGB', (new_w, new_h), (128, 128, 128))
    padded_img.paste(image, (0, 0))
    return padded_img, (w, h), (new_w, new_h)

def parse_and_recover_boxes_direct(text, orig_size, pad_size):
    orig_w, orig_h = orig_size
    pad_w, pad_h = pad_size
    preds = []
    pattern = r'\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]'
    matches = re.findall(pattern, text)
    for m in matches:
        nx0, ny0, nx1, ny1 = map(float, m)
        px0 = min(max((nx0 / 1000) * pad_w, 0), orig_w)
        py0 = min(max((ny0 / 1000) * pad_h, 0), orig_h)
        px1 = min(max((nx1 / 1000) * pad_w, 0), orig_w)
        py1 = min(max((ny1 / 1000) * pad_h, 0), orig_h)
        preds.append([px0, py0, px1, py1])
    return np.array(preds) if preds else np.zeros((0, 4))

# --- 2. 批处理推理封装 ---

def run_internvl_batch(model, processor, batch_imgs, prompts, max_tokens=256):
    batch_chat = []
    for p in prompts:
        msg = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": p}]}]
        batch_chat.append(processor.apply_chat_template(msg, add_generation_prompt=True))
    
    inputs = processor(text=batch_chat, images=batch_imgs, return_tensors="pt", padding=True).to(model.device, dtype=torch.bfloat16)
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
        responses = processor.batch_decode(generated_ids[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return responses

# --- 3. 评测主逻辑 ---

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_dir", type=str, default="/root/autodl-tmp/img_data/target")
    parser.add_argument("--img_root", type=str, default="/root/autodl-tmp/img_data/images")
    parser.add_argument("--model_id", type=str, default="/root/autodl-tmp/local_weights/OpenGVLab/InternVL3_5-8B-hf")
    parser.add_argument("--batch_size", type=int, default=1)
    args = parser.parse_args()

    processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(args.model_id, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True).eval()

    json_files = [f for f in os.listdir(args.json_dir) if f.endswith('.json')]
    dimension_results = {}

    for json_name in sorted(json_files):
        dim_name = os.path.splitext(json_name)[0]
        json_path = os.path.join(args.json_dir, json_name)
        with open(json_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)

        tasks = [item for item in dataset if not item.get("no_target", False)]
        if not tasks: continue

        dim_tp, dim_fp, dim_fn = 0, 0, 0
        print(f"\n>>> 正在评测维度 (多任务一致性模式): {dim_name}")

        for i in tqdm(range(0, len(tasks), args.batch_size)):
            batch = tasks[i : i + args.batch_size]
            batch_imgs, meta_list = [], []
            
            for item in batch:
                raw_img = Image.open(os.path.join(args.img_root, item["file_name"])).convert("RGB")
                padded_img, orig_sz, pad_sz = preprocess_to_32x(raw_img)
                batch_imgs.append(padded_img)
                meta_list.append({"orig_sz": orig_sz, "pad_sz": pad_sz})

            # --- 步骤 1: 同时进行计数和初次检测 ---
            count_prompts = [f"How many instances of '<ref>{item['refer_text']}</ref>' are in this image? Output only the number." for item in batch]
            det_prompts = [f"Please provide the bounding box coordinates of: <ref>{item['refer_text']}</ref>" for item in batch]

            count_res = run_internvl_batch(model, processor, batch_imgs, count_prompts, max_tokens=10)
            det_res = run_internvl_batch(model, processor, batch_imgs, det_prompts, max_tokens=256)

            final_preds_list = []
            arb_indices, arb_imgs, arb_prompts = [], [], []

            for idx, (c_text, d_text) in enumerate(zip(count_res, det_res)):
                # 提取数字
                c_match = re.search(r'\d+', c_text)
                exp_count = int(c_match.group()) if c_match else 1 # 默认为1
                
                initial_preds = parse_and_recover_boxes_direct(d_text, meta_list[idx]["orig_sz"], meta_list[idx]["pad_sz"])
                final_preds_list.append(initial_preds)

                # --- 步骤 2: 一致性检查 ---
                if len(initial_preds) != exp_count:
                    arb_indices.append(idx)
                    arb_imgs.append(batch_imgs[idx])
                    arb_q = (f"Conflict detected: I previously thought there were {exp_count} instances, "
                             f"but only found {len(initial_preds)} boxes. Please re-examine the image carefully "
                             f"and provide the final precise bounding boxes for: <ref>{batch[idx]['refer_text']}</ref>")
                    arb_prompts.append(arb_q)

            # --- 步骤 3: 二次检查 (冲突仲裁) ---
            if arb_indices:
                arb_res = run_internvl_batch(model, processor, arb_imgs, arb_prompts, max_tokens=300)
                for sub_idx, res_text in enumerate(arb_res):
                    real_idx = arb_indices[sub_idx]
                    new_preds = parse_and_recover_boxes_direct(res_text, meta_list[real_idx]["orig_sz"], meta_list[real_idx]["pad_sz"])
                  
                    if len(new_preds) > 0:
                        final_preds_list[real_idx] = new_preds

            # 统计结果
            for idx, preds in enumerate(final_preds_list):
                gts = np.array(batch[idx]["gt_bboxes"])
                tp = solve_matching(preds, gts)
                dim_tp += tp
                dim_fp += (len(preds) - tp)
                dim_fn += (len(gts) - tp)

        dimension_results[dim_name] = {"tp": dim_tp, "fp": dim_fp, "fn": dim_fn}

    # --- 打印最终报告 ---
    print("\n" + "="*90)
    print(f"{'维度名称 (Dimension)':<30} | {'GT数':<6} | {'精确率':<10} | {'召回率':<10} | {'F1-Score'}")
    print("-" * 90)

    total_tp, total_fp, total_fn = 0, 0, 0
    for dim, res in dimension_results.items():
        tp, fp, fn = res["tp"], res["fp"], res["fn"]
        gt_count = tp + fn
        prec = tp / (tp + fp + 1e-6)
        recall = tp / (tp + fn + 1e-6)
        f1 = 2 * prec * recall / (prec + recall + 1e-6)
        
        print(f"{dim:<30} | {gt_count:<6} | {prec:9.2%} | {recall:9.2%} | {f1:.4f}")
        
        total_tp += tp
        total_fp += fp
        total_fn += fn

    overall_prec = total_tp / (total_tp + total_fp + 1e-6)
    overall_recall = total_tp / (total_tp + total_fn + 1e-6)
    overall_f1 = 2 * overall_prec * overall_recall / (overall_prec + overall_recall + 1e-6)

    print("-" * 90)
    print(f"{'OVERALL (Micro)':<30} | {total_tp + total_fn:<6} | {overall_prec:9.2%} | {overall_recall:9.2%} | {overall_f1:.4f}")
    print("="*90)

if __name__ == "__main__":
    main()
