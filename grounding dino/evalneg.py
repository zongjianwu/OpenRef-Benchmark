import argparse
import os
import json
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
import math

# 假设你的环境已经配置好 GroundingDINO
from groundingdino.util.slconfig import SLConfig
from groundingdino.models import build_model
from groundingdino.util.utils import clean_state_dict
import groundingdino.datasets.transforms as T

def get_all_confidences(model, image_path, text_prompt, box_threshold, device):
    """
    获取所有超过阈值的预测框的置信度列表
    """
    image_pil = Image.open(image_path).convert("RGB")
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    image, _ = transform(image_pil, None)
    
    caption = text_prompt.lower().strip() + "."
    with torch.no_grad():
        outputs = model(image[None].to(device), captions=[caption])
    
    # GroundingDINO 输出 logits, 经过 sigmoid 得到置信度
    logits = outputs["pred_logits"].sigmoid()[0]  # [num_queries, num_classes]
    confidences = logits.max(dim=1)[0] # 获取每个 query 最高的类别分数
    
    # 筛选超过阈值的框
    filt_mask = confidences > box_threshold
    final_confs = confidences[filt_mask].cpu().numpy().tolist()
    
    return final_confs

def main():
    parser = argparse.ArgumentParser(description="Confidence-Product NSR Evaluation")
    parser.add_argument("--json_path", type=str, required=True)
    parser.add_argument("--img_dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default="MODEL/groundingdino_swinb_cogcoor.pth")
    parser.add_argument("--config", type=str, default="config/cfg_odvg.py")
    parser.add_argument("--box_thresh", type=float, default=0.3, help="Confidence threshold")
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

    metrics = {
        "absolute": {"strict_rej": 0, "cp_nsr_score": 0.0, "count": 0},
        "relative": {"strict_rej": 0, "cp_nsr_score": 0.0, "count": 0}
    }

    print(f"Evaluating {len(data)} samples using CP-NSR logic...")

    for item in tqdm(data):
        img_path = os.path.join(args.img_dir, item["image"])
        if not os.path.exists(img_path): continue
            
        query = item["negative"]
        m_type = "absolute" if item.get("relation_type") == "none" else "relative"
        metrics[m_type]["count"] += 1
        
        # 获取所有幻觉框的置信度 c_k
        conf_list = get_all_confidences(model, img_path, query, args.box_thresh, device)

        if len(conf_list) == 0:
            # 成功拒识：Strict NSR +1, Score = 1.0
            metrics[m_type]["strict_rej"] += 1
            sample_score = 1.0
        else:
            # 产生幻觉：计算连乘惩罚 Score = Product(1 - c_k)
            # 框越多、越自信，分数下降越快
            sample_score = 1.0
            for c_k in conf_list:
                sample_score *= (1.0 - c_k)
        
        metrics[m_type]["cp_nsr_score"] += sample_score

    # --- 报告生成 ---
    print("\n" + "="*60)
    print("NEGATIVE SAMPLE REJECTION (CP-NSR) REPORT")
    print("Formula: Score = Product of (1 - conf_k) if hallucinated")
    print("="*60)
    
    for t in ["absolute", "relative"]:
        m = metrics[t]
        if m["count"] > 0:
            avg_strict = m["strict_rej"] / m["count"]
            avg_cp_nsr = m["cp_nsr_score"] / m["count"]
            print(f"[{t.upper()}] Count: {m['count']}")
            print(f"  Strict Rejection Acc: {avg_strict:.4%}")
            print(f"  CP-Weighted NSR:      {avg_cp_nsr:.4f}")
            print("-" * 30)
    print("="*60)

if __name__ == "__main__":
    main()