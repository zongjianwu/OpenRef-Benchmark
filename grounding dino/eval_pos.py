import argparse
import os
import json
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from scipy.optimize import linear_sum_assignment
import math

# 导入 DINO 相关组件
from groundingdino.util.slconfig import SLConfig
from groundingdino.models import build_model
from groundingdino.util.utils import clean_state_dict
import groundingdino.datasets.transforms as T
from groundingdino.util import box_ops

def solve_matching(pred_boxes, gt_boxes, iou_threshold=0.5):
    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return 0
    res = box_ops.box_iou(pred_boxes, gt_boxes)
    iou_matrix = res[0] if isinstance(res, tuple) else res
    iou_matrix_np = iou_matrix.cpu().numpy()
    
    cost_matrix = -iou_matrix_np
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    tp = 0
    for r, c in zip(row_ind, col_ind):
        if iou_matrix_np[r, c] >= iou_threshold:
            tp += 1
    return tp

def parse_label_studio_bbox(proposal):
    W, H = proposal["original_width"], proposal["original_height"]
    x, y = proposal["x"] * W / 100, proposal["y"] * H / 100
    w, h = proposal["width"] * W / 100, proposal["height"] * H / 100
    return [x, y, x + w, y + h]

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
    final_confs = confidences[filt_mask]
    
    return final_boxes.cpu(), final_confs.cpu()

def main():
    parser = argparse.ArgumentParser(description="REC Multi-Scenario Evaluation")
    parser.add_argument("--json_path", type=str, required=True)
    parser.add_argument("--img_dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default="config/cfg_odvg.py")
    parser.add_argument("--box_thresh", type=float, default=0.3)
    parser.add_argument("--iou_thresh", type=float, default=0.5)
    parser.add_argument("--alpha", type=float, default=1.0)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = SLConfig.fromfile(args.config)
    cfg.device = device
    model = build_model(cfg)
    
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    model.to(device).eval()
    
    with open(args.json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 统计容器
    pos_tp, pos_fp, pos_fn = 0, 0, 0
    all_sample_f1 = []
    nsr_strict_sum, nsr_weighted_sum = 0, 0
    neg_count = 0

    print(f"Evaluating {len(data)} samples (Dual-Prompt Mode)...")
    for item in tqdm(data):
        img_path = os.path.join(args.img_dir, item["image"])
        
        # --- 1. 正样本评测 (使用 Positive Prompt) ---
        p_query = item["positive"]
        p_boxes, _ = get_prediction_with_conf(model, img_path, p_query, args.box_thresh, device)
        
        gt_boxes = torch.tensor([parse_label_studio_bbox(p) for p in item["proposal"]])
        tp_i = solve_matching(p_boxes, gt_boxes, args.iou_thresh)
        
        pos_tp += tp_i
        pos_fp += (len(p_boxes) - tp_i)
        pos_fn += (len(gt_boxes) - tp_i)
        
        f1_i = (2 * tp_i) / (2 * tp_i + (len(p_boxes)-tp_i) + (len(gt_boxes)-tp_i)) if (2 * tp_i + len(p_boxes)-tp_i + len(gt_boxes)-tp_i) > 0 else 0
        all_sample_f1.append(f1_i)

        # --- 2. 负样本评测 (使用 Negative Prompt) ---
        n_query = item.get("negative")
        if n_query:
            neg_count += 1
            n_boxes, n_confs = get_prediction_with_conf(model, img_path, n_query, args.box_thresh, device)
            num_n_preds = len(n_boxes)
            
            # Strict NSR: 无输出即为满分
            if num_n_preds == 0:
                nsr_strict_sum += 1
            
            # Weighted NSR: 结合置信度和数量的惩罚
            max_c = n_confs.max().item() if num_n_preds > 0 else 0
            nsr_weighted_sum += math.exp(-args.alpha * (num_n_preds + max_c))

    # 输出结果报告
    precision = pos_tp / (pos_tp + pos_fp) if (pos_tp + pos_fp) > 0 else 0
    recall = pos_tp / (pos_tp + pos_fn) if (pos_tp + pos_fn) > 0 else 0
    
    print("\n" + "="*50)
    print(f"REC EVALUATION REPORT")
    print("="*50)
    print(f"[Positive Tasks (Query: '{item['positive'] if 'item' in locals() else '...'}')]")
    print(f"  Precision (Global): {precision:.4f}")
    print(f"  Recall    (Global): {recall:.4f}")
    print(f"  Mean F1   (Sample): {np.mean(all_sample_f1):.4f}")
    
    if neg_count > 0:
        print(f"\n[Negative Tasks (Query: '{item['negative'] if 'item' in locals() else '...'}')]")
        print(f"  Strict NSR:         {nsr_strict_sum / neg_count:.4f}")
        print(f"  Weighted NSR:       {nsr_weighted_sum / neg_count:.4f}")
    print("="*50)

if __name__ == "__main__":
    main()