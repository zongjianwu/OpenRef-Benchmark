import argparse
import os
import json
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from scipy.optimize import linear_sum_assignment
import math

# 假设 groundingdino 已在环境变量或当前目录下
from groundingdino.util.slconfig import SLConfig
from groundingdino.models import build_model
from groundingdino.util.utils import clean_state_dict
import groundingdino.datasets.transforms as T
from groundingdino.util import box_ops

# --- 1. 工具函数 ---

def solve_matching_by_size(pred_boxes, gt_boxes, img_w, img_h, iou_threshold=0.5):
    """按目标大小分类统计 TP 和 GT 数量"""
    stats = {"tiny": [0, 0], "small": [0, 0], "medium": [0, 0], "large": [0, 0]}
    num_gt = len(gt_boxes)
    if num_gt == 0: return stats

    img_area = img_w * img_h
    gt_types = []
    for i in range(num_gt):
        box = gt_boxes[i]
        area = (box[2] - box[0]) * (box[3] - box[1])
        ratio = area / img_area
        if ratio < 0.01: obj_type = "tiny"
        elif ratio < 0.10: obj_type = "small"
        elif ratio < 0.50: obj_type = "medium"
        else: obj_type = "large"
        gt_types.append(obj_type)
        stats[obj_type][1] += 1 

    if len(pred_boxes) == 0: return stats

    res = box_ops.box_iou(pred_boxes, gt_boxes)
    iou_matrix_np = (res[0] if isinstance(res, tuple) else res).cpu().numpy()
    row_ind, col_ind = linear_sum_assignment(-iou_matrix_np)
    
    for r, c in zip(row_ind, col_ind):
        if iou_matrix_np[r, c] >= iou_threshold:
            obj_type = gt_types[c]
            stats[obj_type][0] += 1
    return stats

def parse_label_studio_bbox(proposal):
    W, H = proposal["original_width"], proposal["original_height"]
    return [proposal["x"] * W / 100, proposal["y"] * H / 100, 
            (proposal["x"] + proposal["width"]) * W / 100, (proposal["y"] + proposal["height"]) * H / 100]

def get_prediction_with_conf(model, image_path, text_prompt, box_threshold, device):
    image_pil = Image.open(image_path).convert("RGB")
    W, H = image_pil.size
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    image, _ = transform(image_pil, None)
    caption = text_prompt.lower().strip()
    if not caption.endswith("."): caption += "."
    with torch.no_grad():
        outputs = model(image[None].to(device), captions=[caption])
    logits = outputs["pred_logits"].sigmoid()[0]
    boxes = outputs["pred_boxes"][0]
    confidences = logits.max(dim=1)[0]
    filt_mask = confidences > box_threshold
    final_boxes = box_ops.box_cxcywh_to_xyxy(boxes[filt_mask]) * torch.Tensor([W, H, W, H]).to(device)
    return final_boxes.cpu(), confidences[filt_mask].cpu(), W, H

# --- 2. 主逻辑 ---

def main():
    parser = argparse.ArgumentParser(description="GroundingDINO Multi-Scale Eval")
    parser.add_argument("--test_root", type=str, required=True, help="根目录路径")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--box_thresh", type=float, default=0.3)
    parser.add_argument("--iou_thresh", type=float, default=0.5)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = SLConfig.fromfile(args.config)
    cfg.device = device
    model = build_model(cfg)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    model.to(device).eval()

    # 1. 自动识别维度文件夹 (排除 'none')
    all_dims = [d for d in os.listdir(args.test_root) 
                if os.path.isdir(os.path.join(args.test_root, d)) and d != "none"]
    
    # 2. 初始化全局统计
    global_total = {
        "tiny": {"tp": 0, "gt": 0},
        "small": {"tp": 0, "gt": 0},
        "medium": {"tp": 0, "gt": 0},
        "large": {"tp": 0, "gt": 0}
    }

    # 3. 遍历每个维度进行评测
    for dim in all_dims:
        json_path = os.path.join(args.test_root, f"{dim}_test_labels.json")
        img_dir = os.path.join(args.test_root, dim)
        if not os.path.exists(json_path): continue

        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 仅处理 positive 样本
        tasks = [item for item in data if item.get("positive")]

        for item in tqdm(tasks, desc=f"Eval {dim}"):
            img_path = os.path.join(img_dir, item["image"])
            if not os.path.exists(img_path): continue
            
            p_query = item["positive"]
            p_boxes, _, img_w, img_h = get_prediction_with_conf(model, img_path, p_query, args.box_thresh, device)
            gt_boxes = torch.tensor([parse_label_studio_bbox(p) for p in item["proposal"]])
            
            match_res = solve_matching_by_size(p_boxes, gt_boxes, img_w, img_h, args.iou_thresh)
            
            for sz in global_total:
                global_total[sz]["tp"] += match_res[sz][0]
                global_total[sz]["gt"] += match_res[sz][1]

    # --- 4. 输出最终报表 ---
    print("\n" + "="*110)
    print(f"{'Model: GroundingDINO':<30} | {'Tiny':<10} | {'Small':<10} | {'Medium':<10} | {'Large':<10} | {'mRecall'}")
    print("-" * 110)
    
    recalls = []
    row = [f"{args.checkpoint.split('/')[-1][:30]:<30}"]
    
    for sz in ["tiny", "small", "medium", "large"]:
        tp = global_total[sz]["tp"]
        gt = global_total[sz]["gt"]
        if gt > 0:
            r = tp / gt
            recalls.append(r)
            row.append(f"{r:9.2%}")
        else:
            row.append(f"{'N/A':>9}")

    avg_recall = np.mean(recalls) if recalls else 0
    print(" | ".join(row) + f" | {avg_recall:.2%}")
    print("="*110)

if __name__ == "__main__":
    main()