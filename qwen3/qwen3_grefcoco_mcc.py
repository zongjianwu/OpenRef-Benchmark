import argparse
import os
import json
import torch
import numpy as np
import re
import glob
from tqdm import tqdm
from PIL import Image
from scipy.optimize import linear_sum_assignment
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

# --- 1. 核心工具函数 ---

def solve_matching(pred_boxes, gt_boxes, iou_threshold=0.5):
    """使用匈牙利算法进行正样本匹配"""
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
    fp = len(pred_boxes) - tp
    fn = len(gt_boxes) - tp
    return tp, fp, fn

def parse_boxes(text, img_w, img_h):
    """解析坐标 [xmin, ymin, xmax, ymax] 并反归一化"""
    preds = []
    pattern = r"\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]"
    matches = re.findall(pattern, text)
    for m in matches:
        preds.append([
            int(m[0]) * img_w / 1000, int(m[1]) * img_h / 1000,
            int(m[2]) * img_w / 1000, int(m[3]) * img_h / 1000
        ])
    return preds

class MetricsAccumulator:
    def __init__(self):
        self.tp, self.fp, self.fn = 0, 0, 0
        self.f1_list = []
        self.count = 0

    def update(self, tp, fp, fn):
        self.tp += tp
        self.fp += fp
        self.fn += fn
        f1 = (2 * tp) / (2 * tp + fp + fn + 1e-6) if (tp + fp + fn) > 0 else 0
        self.f1_list.append(f1)
        self.count += 1

    def get_results(self):
        p = self.tp / (self.tp + self.fp + 1e-6)
        r = self.tp / (self.tp + self.fn + 1e-6)
        mean_f1 = np.mean(self.f1_list) if self.f1_list else 0
        return p, r, mean_f1

def run_qwen_inference(model, processor, images, prompts, max_tokens=256):
    """通用的批量推理封装"""
    messages = [
        [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": q}]}]
        for img, q in zip(images, prompts)
    ]
    
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    vision_res = process_vision_info(messages)
    image_inputs, video_inputs = vision_res[0], vision_res[1]
    
    inputs = processor(
        text=text, 
        images=image_inputs, 
        videos=video_inputs, 
        padding=True, 
        return_tensors="pt"
    ).to(model.device)

    with torch.inference_mode():
        generated_ids = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        return processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)

# --- 3. 主程序：多任务一致性约束逻辑 ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="/root/autodl-tmp/img_data")
    parser.add_argument("--model_path", type=str, default="/root/autodl-tmp/Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    # 加载模型与处理器
    print(f"Loading Model: {args.model_path}")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    ).eval()
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    processor.tokenizer.padding_side = "left"

    json_files = glob.glob(os.path.join(args.data_root, "*.json")) + \
                 glob.glob(os.path.join(args.data_root, "none/*.json"))
    
    img_dir = os.path.join(args.data_root, "images")
    summary_results = []

    for json_path in json_files:
        file_name = os.path.basename(json_path)
        print(f"\n>>> Processing: {file_name} (Task Consistency Mode)")
        
        with open(json_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)

        accu = MetricsAccumulator()
        correct_refusal = 0
        is_none_task = any(k in file_name for k in ["none", "negative"])

        for i in tqdm(range(0, len(dataset), args.batch_size)):
            batch_data = dataset[i : i + args.batch_size]
            imgs, valid_items = [], []
            for item in batch_data:
                full_img_path = os.path.join(img_dir, item["file_name"])
                if os.path.exists(full_img_path):
                    imgs.append(Image.open(full_img_path).convert("RGB"))
                    valid_items.append(item)
            
            if not imgs: continue

            # --- 阶段 A: 获取独立计数与初次检测 ---
            count_qs = [f"How many '{item['refer_text']}' are in this image? Output only the number." for item in valid_items]
            det_qs = [f"Detect <{item['refer_text']}> and output all bounding boxes in [[xmin,ymin,xmax,ymax]] format." for item in valid_items]

            count_res = run_qwen_inference(model, processor, imgs, count_qs, max_tokens=10)
            det_res = run_qwen_inference(model, processor, imgs, det_qs, max_tokens=256)

            final_preds_list = []
            arb_indices, arb_imgs, arb_prompts = [], [], []

            for idx, (c_resp, d_resp) in enumerate(zip(count_res, det_res)):
                # 1. 提取预测数量
                c_match = re.search(r'\d+', c_resp)
                exp_count = int(c_match.group()) if c_match else 1
                
                # 2. 解析初次检测框
                w, h = imgs[idx].size
                initial_preds = parse_boxes(d_resp, w, h)
                final_preds_list.append(initial_preds)

                # 3. 检查冲突
                if not is_none_task and len(initial_preds) != exp_count:
                    arb_indices.append(idx)
                    arb_imgs.append(imgs[idx])
                    # 提示词强调：指出差异，要求模型重新思考
                    arb_q = (f"Wait, you said there are {exp_count} instances of '{valid_items[idx]['refer_text']}', "
                             f"but you only provided {len(initial_preds)} boxes. "
                             f"Please re-examine the image carefully and provide ALL the correct bounding boxes.")
                    arb_prompts.append(arb_q)

            # --- 阶段 B: 冲突仲裁 (二次重审) ---
            if arb_indices:
                arb_res = run_qwen_inference(model, processor, arb_imgs, arb_prompts, max_tokens=300)
                for sub_idx, res_text in enumerate(arb_res):
                    real_idx = arb_indices[sub_idx]
                    w, h = imgs[real_idx].size
                    refined_preds = parse_boxes(res_text, w, h)
                    
                    if len(refined_preds) > 0:
                        final_preds_list[real_idx] = refined_preds

            # --- 阶段 C: 最终统计 ---
            for idx, preds in enumerate(final_preds_list):
                if is_none_task:
                    accu.count += 1
                    if len(preds) == 0: correct_refusal += 1
                else:
                    gts = valid_items[idx].get("gt_bboxes", [])
                    tp, fp, fn = solve_matching(preds, gts)
                    accu.update(tp, fp, fn)

        if is_none_task:
            acc = correct_refusal / (accu.count + 1e-6)
            summary_results.append({"file": file_name, "type": "Negative", "p": acc, "r": "-", "f1": "-"})
        else:
            p, r, f1 = accu.get_results()
            summary_results.append({"file": file_name, "type": "Positive", "p": p, "r": r, "f1": f1})

    print("\n" + "="*95)
    print(f"{'JSON File Name':<35} | {'Type':<10} | {'Prec/Acc':<10} | {'Recall':<10} | {'Mean F1':<10}")
    print("-" * 95)
    for res in summary_results:
        p_val = f"{res['p']:.4f}" if isinstance(res['p'], float) else res['p']
        r_val = f"{res['r']:.4f}" if isinstance(res['r'], float) else res['r']
        f1_val = f"{res['f1']:.4f}" if isinstance(res['f1'], float) else res['f1']
        print(f"{res['file']:<35} | {res['type']:<10} | {p_val:<10} | {r_val:<10} | {f1_val:<10}")
    print("="*95)

if __name__ == "__main__":
    main()
