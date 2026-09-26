import argparse
import os
import json
import torch
import numpy as np
import re
from tqdm import tqdm
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

# 设置环境变量防止 libgomp 报错
os.environ["OMP_NUM_THREADS"] = "1"

def get_per_box_conf(gen_ids, transition_scores, processor):
    probs = torch.exp(transition_scores)
    batch_box_confs = []
    for i in range(gen_ids.shape[0]):
        sample_ids = gen_ids[i]
        sample_probs = probs[i]
        digit_probs = []
        for tid, p in zip(sample_ids, sample_probs):
            token_text = processor.decode([tid]).strip()
            if token_text.isdigit():
                digit_probs.append(p.item())
        box_confs = []
        for j in range(0, len(digit_probs) // 4 * 4, 4):
            avg_conf = np.mean(digit_probs[j : j+4])
            box_confs.append(avg_conf)
        batch_box_confs.append(box_confs)
    return batch_box_confs

def parse_qwen_output(output_text):
    text_lower = output_text.lower()
    negative_words = ["no ", "none", "not found", "isn't", "unable", "not detect", "nothing"]
    semantic_rejected = any(word in text_lower for word in negative_words)
    coords = re.findall(r'(\d+),\s*(\d+),\s*(\d+),\s*(\d+)', output_text)
    is_actually_rejected = (semantic_rejected or "[]" in output_text) and (len(coords) == 0)
    return coords, is_actually_rejected

def evaluate_single_dimension(model, processor, json_path, base_dir, batch_size):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    metrics = {"correct_rej": 0, "weighted_score": 0.0, "count": 0}
    
    # 调试：打印第一个样本的预期路径
    if len(data) > 0:
        test_path = os.path.join(base_dir, data[0]["image"])
        print(f"\n[DEBUG] 正在校验维度 {os.path.basename(json_path)}，首张图片预期路径: {test_path}")

    for i in tqdm(range(0, len(data), batch_size), desc=f"Processing {os.path.basename(json_path)}", leave=False):
        batch_items = data[i : i + batch_size]
        batch_imgs, batch_texts, active_items = [], [], []

        for item in batch_items:
            # 路径纠错逻辑
            img_rel_path = item["image"]
            img_path = os.path.join(base_dir, img_rel_path)
            
            # 如果路径不通，尝试在该维度对应的子文件夹下寻找
            if not os.path.exists(img_path):
                dim_subfolder = os.path.basename(json_path).replace("_test_labels.json", "")
                img_path = os.path.join(base_dir, dim_subfolder, os.path.basename(img_rel_path))

            if not os.path.exists(img_path):
                continue
            
            try:
                img = Image.open(img_path).convert("RGB")
                batch_imgs.append(img)
                active_items.append(item)
                prompt = f"Detect '{item['negative']}'. If it exists, return JSON format with 'bbox_2d'. If not, return 'none'."
                text = processor.apply_chat_template(
                    [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": prompt}]}],
                    tokenize=False, add_generation_prompt=True
                )
                batch_texts.append(text)
            except:
                continue

        if not batch_imgs: continue

        inputs = processor(text=batch_texts, images=batch_imgs, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=128, return_dict_in_generate=True, output_scores=True)
        
        gen_ids = outputs.sequences[:, inputs.input_ids.shape[1]:]
        responses = processor.batch_decode(gen_ids, skip_special_tokens=True)
        transition_scores = model.compute_transition_scores(outputs.sequences, outputs.scores, normalize_logits=True)
        batch_box_confs = get_per_box_conf(gen_ids, transition_scores, processor)

        for idx, _ in enumerate(active_items):
            metrics["count"] += 1
            _, is_rej = parse_qwen_output(responses[idx])
            box_confs = batch_box_confs[idx]
            if is_rej:
                metrics["correct_rej"] += 1
                metrics["weighted_score"] += 1.0
            else:
                sample_score = 1.0
                for c_k in box_confs:
                    sample_score *= (1.0 - c_k)
                metrics["weighted_score"] += (sample_score if box_confs else 0.5)

    return metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_dir", type=str, default="/root/autodl-tmp/test2")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2-VL-7B-Instruct")
    parser.add_argument("--batch_size", type=int, default=1)
    args = parser.parse_args()

    # 显式使用 dtype 消除警告
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model_id, dtype=torch.bfloat16, device_map="auto"
    ).eval()
    processor = AutoProcessor.from_pretrained(args.model_id, use_fast=False)
    processor.tokenizer.padding_side = 'left'

    json_files = sorted([f for f in os.listdir(args.test_dir) if f.endswith("_test_labels.json")])
    
    if not json_files:
        print(f"Error: 在目录 {args.test_dir} 下未找到 JSON 文件！")
        return

    results_table = []
    total_samples, total_rej, total_weighted = 0, 0, 0.0

    for j_file in json_files:
        dim_name = j_file.replace("_test_labels.json", "")
        res = evaluate_single_dimension(model, processor, os.path.join(args.test_dir, j_file), args.test_dir, args.batch_size)
        
        if res["count"] > 0:
            results_table.append({
                "dim": dim_name, "count": res["count"],
                "acc": res["correct_rej"] / res["count"],
                "score": res["weighted_score"] / res["count"]
            })
            total_samples += res["count"]
            total_rej += res["correct_rej"]
            total_weighted += res["weighted_score"]
        else:
            print(f"Warning: 维度 {dim_name} 有效样本数为 0，请检查图片路径是否匹配！")

    print("\n" + "="*85)
    print(f"{'Dimension':<20} | {'Count':<8} | {'Hard Rej Acc':<15} | {'CP-Weighted Score':<18}")
    print("-" * 85)
    for r in results_table:
        print(f"{r['dim']:<20} | {r['count']:<8} | {r['acc']:<15.4%} | {r['score']:<18.4f}")
    if total_samples > 0:
        print("-" * 85)
        print(f"{'OVERALL MEAN':<20} | {total_samples:<8} | {total_rej/total_samples:<15.4%} | {total_weighted/total_samples:<18.4f}")
    print("="*85 + "\n")

if __name__ == "__main__":
    main()
