import argparse
import os
import json
import torch
import numpy as np
import re
from tqdm import tqdm
from PIL import Image
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# --- 1. 工具函数 ---

def get_box_confidence(gen_ids, transition_scores, processor):
    """提取 Qwen 架构中数字坐标 Token 的平均概率"""
    probs = torch.exp(transition_scores)
    batch_box_confs = []
    for i in range(gen_ids.shape[0]):
        sample_ids = gen_ids[i]
        sample_probs = probs[i]
        digit_probs = [p.item() for tid, p in zip(sample_ids, sample_probs) if processor.decode([tid]).strip().isdigit()]
        # 每 4 个数字构成一个框
        box_confs = [np.mean(digit_probs[j : j+4]) for j in range(0, len(digit_probs) // 4 * 4, 4)]
        batch_box_confs.append(box_confs)
    return batch_box_confs

def parse_qwen_output(output_text, img_w, img_h):
    """解析输出并返回坐标及拒识状态"""
    text_lower = output_text.lower()
    negative_words = ["no ", "none", "not found", "isn't", "unable", "not detect", "nothing"]
    is_rejected = any(word in text_lower for word in negative_words)
    
    preds = []
    coords = re.findall(r'(\d+),\s*(\d+),\s*(\d+),\s*(\d+)', output_text)
    for c in coords:
        b = [int(x) for x in c]
        preds.append([b[0]*img_w/1000, b[1]*img_h/1000, b[2]*img_w/1000, b[3]*img_h/1000])
    
    # 只要产生框，就不视为 Hard Rejection
    actual_rej = is_rejected and (len(preds) == 0)
    return np.array(preds), actual_rej

def run_qwen_inference(model, processor, batch_messages, max_tokens=256):
    texts = [processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in batch_messages]
    image_inputs, video_inputs = process_vision_info(batch_messages)
    inputs = processor(text=texts, images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=max_tokens, do_sample=False,
            return_dict_in_generate=True, output_scores=True
        )
    
    gen_ids = outputs.sequences[:, inputs.input_ids.shape[1]:]
    responses = processor.batch_decode(gen_ids, skip_special_tokens=True)
    t_scores = model.compute_transition_scores(outputs.sequences, outputs.scores, normalize_logits=True)
    box_confs = get_box_confidence(gen_ids, t_scores, processor)
    return responses, box_confs

# --- 2. 主评测流程 ---

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_root", type=str, default="test2")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen3-VL-2B-Instruct")
    args = parser.parse_args()

    model = Qwen3VLForConditionalGeneration.from_pretrained(args.model_id, torch_dtype=torch.bfloat16, device_map="auto").eval()
    processor = AutoProcessor.from_pretrained(args.model_id)
    processor.tokenizer.padding_side = "left"

    all_dims = [d for d in os.listdir(args.test_root) if os.path.isdir(os.path.join(args.test_root, d)) and not d.startswith('.')]
    all_results = {}

    for dim in all_dims:
        json_path = os.path.join(args.test_root, f"{dim}_test_labels.json")
        img_dir = os.path.join(args.test_root, dim)
        if not os.path.exists(json_path): continue

        with open(json_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)

        task_queue = [{"img": os.path.join(img_dir, item["image"]), "query": item["negative"]} 
                      for item in raw_data if item.get("negative")]

        # 统计变量
        dim_metrics = {"count": 0, "hard_rej_count": 0, "n3r_score_sum": 0.0}
        print(f"\n>>> 维度: {dim} | 样本量: {len(task_queue)}")

        for i in tqdm(range(0, len(task_queue), args.batch_size), desc=f"Scanning {dim}"):
            batch_tasks = task_queue[i : i + args.batch_size]
            
            # --- Pass 1 ---
            msgs_r1 = [[{"role": "user", "content": [{"type": "image", "image": t["img"]}, {"type": "text", "text": f"Detect '{t['query']}'. Return bboxes or 'none'."}]}] for t in batch_tasks]
            res_r1, _ = run_qwen_inference(model, processor, msgs_r1)

            # --- Pass 2 (二次推理) ---
            msgs_r2 = []
            for idx, t in enumerate(batch_tasks):
                msgs_r2.append([
                    {"role": "user", "content": [{"type": "image", "image": t["img"]}, {"type": "text", "text": f"Detect '{t['query']}'. Return bboxes or 'none'."}]},
                    {"role": "assistant", "content": [{"type": "text", "text": res_r1[idx]}]},
                    {"role": "user", "content": [{"type": "text", "text": "Are you sure? Re-examine carefully. If the object isn't there, say 'none'. Otherwise, give bboxes:"}]}
                ])
            
            res_r2, confs_r2 = run_qwen_inference(model, processor, msgs_r2)

            # --- 结果结算 ---
            for idx in range(len(batch_tasks)):
                img = Image.open(batch_tasks[idx]["img"])
                preds, is_rej = parse_qwen_output(res_r2[idx], img.size[0], img.size[1])
                
                dim_metrics["count"] += 1
                if is_rej:
                    dim_metrics["hard_rej_count"] += 1
                    dim_metrics["n3r_score_sum"] += 1.0 # 绝对拒识，得分1.0
                else:
                    # 相对性指标计算: Π (1 - conf_k)
                    sample_score = 1.0
                    if len(confs_r2[idx]) > 0:
                        for c in confs_r2[idx]:
                            sample_score *= (1.0 - c)
                    else:
                        sample_score = 0.5 # 既没说None也没给框的模糊样本
                    dim_metrics["n3r_score_sum"] += sample_score

        all_results[dim] = dim_metrics

    # --- 汇总报告 ---
    print("\n" + "="*85)
    print(f"{'Dimension':<20} | {'Total':<8} | {'NSR (Hard)↑':<15} | {'N3R (Relat.)↑'}")
    print("-" * 85)
    
    t_cnt, t_hr, t_n3r = 0, 0, 0.0
    for dim, m in all_results.items():
        nsr = m['hard_rej_count'] / m['count']
        n3r = m['n3r_score_sum'] / m['count']
        print(f"{dim:<20} | {m['count']:<8} | {nsr:15.2%} | {n3r:.4f}")
        t_cnt += m['count']; t_hr += m['hard_rej_count']; t_n3r += m['n3r_score_sum']

    if t_cnt > 0:
        print("-" * 85)
        print(f"{'OVERALL':<20} | {t_cnt:<8} | {t_hr/t_cnt:15.2%} | {t_n3r/t_cnt:.4f}")
    print("="*85)

if __name__ == "__main__":
    main()
