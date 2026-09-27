import argparse
import os
import json
import torch
import numpy as np
import re
from tqdm import tqdm
from PIL import Image
from scipy.optimize import linear_sum_assignment
from transformers import AutoProcessor, Glm4vForConditionalGeneration

def solve_matching(pred_boxes, gt_boxes, iou_threshold=0.5):
    if len(gt_boxes) == 0: return 0, len(pred_boxes), 0
    if len(pred_boxes) == 0: return 0, 0, len(gt_boxes)
    pred_boxes, gt_boxes = np.array(pred_boxes), np.array(gt_boxes)
    b1, b2 = pred_boxes[:, None, :], gt_boxes[None, :, :]
    inter = np.maximum(0, np.minimum(b1[..., 2:], b2[..., 2:]) - np.maximum(b1[..., :2], b2[..., :2])).prod(-1)
    area1 = (b1[..., 2:] - b1[..., :2]).prod(-1)
    area2 = (b2[..., 2:] - b2[..., :2]).prod(-1)
    iou = inter / (area1 + area2 - inter + 1e-6)
    row_ind, col_ind = linear_sum_assignment(-iou)
    tp = sum(1 for r, c in zip(row_ind, col_ind) if iou[r, c] >= iou_threshold)
    return tp, len(pred_boxes) - tp, len(gt_boxes) - tp

def parse_label_studio_bbox(proposal, img_w, img_h):
    return [
        proposal["x"] * img_w / 100, proposal["y"] * img_h / 100, 
        (proposal["x"] + proposal["width"]) * img_w / 100, (proposal["y"] + proposal["height"]) * img_h / 100
    ]

def parse_glm_special_output(text, img_w, img_h):
    preds = []
    pattern = r"\[\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]\]"
    matches = re.findall(pattern, text)
    for m in matches:
        x1, y1, x2, y2 = map(int, m)
        preds.append([x1 * img_w / 1000, y1 * img_h / 1000, x2 * img_w / 1000, y2 * img_h / 1000])
    return preds

def parse_count(text):
    nums = re.findall(r'\d+', text)
    return int(nums[0]) if nums else 0

def build_image_map(root_dir):
    img_map = {}
    for root, _, files in os.walk(root_dir):
        for f in files:
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                img_map[f] = os.path.join(root, f)
    return img_map

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_root", type=str, default="test2")
    parser.add_argument("--model_path", type=str, default="/root/autodl-tmp/local_weights/ZhipuAI/GLM-4.6V-Flash")
    args = parser.parse_args()

    # --- A. 显存监控：初始状态 ---
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    vram_start = torch.cuda.memory_allocated() / 1024**3

    print(">>> 正在扫描图片目录...")
    image_full_paths = build_image_map(args.test_root)

    # --- B. 加载模型 ---
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    model = Glm4vForConditionalGeneration.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    ).eval()
    
    # 统计静态加载后的显存
    vram_model = torch.cuda.memory_allocated() / 1024**3

    # --- C. 数据加载 ---
    task_queue = []
    for json_name in ["celebrity_test_labels.json", "logo.json"]:
        json_path = os.path.join(args.test_root, json_name)
        if not os.path.exists(json_path): continue
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for item in data:
                img_filename = os.path.basename(item["image"])
                if img_filename in image_full_paths and item.get("positive") and item.get("proposal"):
                    item["abs_path"] = image_full_paths[img_filename]
                    task_queue.append(item)

    print(f">>> 有效任务数量: {len(task_queue)}")

    # --- D. 推理与统计 ---
    stats = {"tp": 0, "fp": 0, "fn": 0, "conflicts": 0, "f1_scores": []}
    max_vram_inference = 0 

    for item in tqdm(task_queue, desc="MCC Evaluating"):
        raw_img = Image.open(item["abs_path"]).convert("RGB")
        w, h = raw_img.size
        obj_name = item['positive']

        def run_model(prompt_text):
            msgs = [{"role": "user", "content": [{"type": "image", "image": raw_img}, {"type": "text", "text": prompt_text}]}]

            inputs = processor.apply_chat_template([msgs], tokenize=True, add_generation_prompt=True, 
                                                  return_dict=True, return_tensors="pt").to(model.device)
            
            with torch.inference_mode():
                outputs = model.generate(**inputs, max_new_tokens=128, do_sample=False)
                # 显存采样：记录运行期间的 Reserved 峰值
                nonlocal max_vram_inference
                max_vram_inference = max(max_vram_inference, torch.cuda.memory_reserved() / 1024**3)
                
                return processor.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

        # Stage 1: Count
        res_cnt = run_model(f"How many '{obj_name}' are there? Answer with a number only.")
        cnt_val = parse_count(res_cnt)

        # Stage 2: Detect
        res_det = run_model(f"Detect every '{obj_name}'. Return [[x1,y1,x2,y2]].")
        preds = parse_glm_special_output(res_det, w, h)

        # Stage 3: MCC
        if cnt_val != len(preds):
            stats["conflicts"] += 1
            res_arb = run_model(f"Conflict: I counted {cnt_val} but found {len(preds)} boxes. Final [[x1,y1,x2,y2]] for '{obj_name}'.")
            preds = parse_glm_special_output(res_arb, w, h)

        # 评估
        gts = [parse_label_studio_bbox(p, w, h) for p in item["proposal"]]
        tp, fp, fn = solve_matching(preds, gts)
        stats["tp"] += tp; stats["fp"] += fp; stats["fn"] += fn
        s_f1 = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
        stats["f1_scores"].append(s_f1)

    # --- E. 最终报告 ---
    vram_peak_system = torch.cuda.max_memory_allocated() / 1024**3
    prec = stats["tp"] / (stats["tp"] + stats["fp"] + 1e-6)
    rec = stats["tp"] / (stats["tp"] + stats["fn"] + 1e-6)
    f1 = np.mean(stats["f1_scores"]) if stats["f1_scores"] else 0

    print("\n" + "="*60)
    print("GLM-4V MCC PERFORMANCE & VRAM REPORT")
    print("-" * 60)
    print(f"Total Samples:      {len(task_queue)}")
    print(f"Conflict Rate:      {stats['conflicts']/len(task_queue):.2%}")
    print(f"Precision:          {prec:.4f}")
    print(f"Recall:             {rec:.4f}")
    print(f"F1 Score:           {f1:.4f}")
    print("-" * 60)
    print(f"VRAM Static Load:   {vram_model:.2f} GB")
    print(f"VRAM Inference Max: {max_vram_inference:.2f} GB") 
    print(f"VRAM System Peak:   {vram_peak_system:.2f} GB")
    print("="*60)

if __name__ == "__main__":
    main()
