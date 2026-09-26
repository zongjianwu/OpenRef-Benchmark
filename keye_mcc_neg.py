import argparse
import os
import json
import torch
import numpy as np
import re
from tqdm import tqdm
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor
from keye_vl_utils import process_vision_info

import warnings
warnings.filterwarnings("ignore")

# --- 1. 核心工具函数 ---

def get_keye_box_confidences(gen_ids, transition_scores, processor):
    """ 提取框坐标 Token 的平均置信度 """
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
        box_confs = [np.mean(digit_probs[j:j+4]) for j in range(0, len(digit_probs)//4*4, 4)]
        batch_box_confs.append(box_confs)
    return batch_box_confs

def parse_keye_nsr_output(text):
    """ 解析输出：提取坐标数量及是否语义拒识 """
    text_lower = text.lower()
    negative_words = ["no ", "none", "not found", "isn't", "unable", "not detect", "nothing", "[]", "zero"]
    semantic_rejected = any(word in text_lower for word in negative_words)
    coords = re.findall(r"\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]", text)
    is_hard_rejected = (len(coords) == 0) and semantic_rejected
    return coords, is_hard_rejected

# --- 2. 推理执行器 ---

def run_keye_inference_full(model, processor, img_path, prompt_text, max_tokens=128):
    """ 封装单次推理逻辑，返回文本结果、序列 IDs 和置信度分值 """
    messages = [{"role": "user", "content": [{"type": "image", "image": img_path}, {"type": "text", "text": prompt_text}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs, mm_processor_kwargs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, 
                       padding=True, return_tensors="pt", **mm_processor_kwargs).to("cuda")
    
    with torch.inference_mode():
        outputs = model.generate(**inputs, max_new_tokens=max_tokens, return_dict_in_generate=True, output_scores=True)
        gen_ids = outputs.sequences[:, inputs.input_ids.shape[1]:]
        response = processor.batch_decode(gen_ids, skip_special_tokens=True)[0]
        transition_scores = model.compute_transition_scores(outputs.sequences, outputs.scores, normalize_logits=True)
        box_confs = get_keye_box_confidences(gen_ids, transition_scores, processor)[0]
    return response, box_confs

# --- 3. 主评测流程 ---

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_root", type=str, default="test3")
    parser.add_argument("--model_path", type=str, default="/root/autodl-tmp/local_weights/Kwai-Keye/Keye-VL-1_5-8B")
    args = parser.parse_args()

    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.bfloat16, 
                                                 trust_remote_code=True, attn_implementation="flash_attention_2").eval().cuda()
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)

    json_files = sorted([f for f in os.listdir(args.test_root) if f.endswith("_test_labels.json")])
    all_results = {}

    for j_file in json_files:
        dim_name = j_file.replace("_test_labels.json", "")
        img_dir = os.path.join(args.test_root, dim_name)
        with open(os.path.join(args.test_root, j_file), 'r', encoding='utf-8') as f:
            dataset = json.load(f)

        print(f"\n>>> 负样本增强评测: [{dim_name.upper()}] | 样本量: {len(dataset)}")
        stats = {"count": 0, "hard_rej": 0, "arb_triggered": 0, "weighted_score": 0.0}

        for item in tqdm(dataset):
            img_path = os.path.join(img_dir, item["image"])
            target = item['negative']
            if not os.path.exists(img_path): continue

            # --- 步骤 1: 独立计数 ---
            cnt_prompt = f"How many <{target}> are there in the image? Answer with a single integer."
            cnt_res, _ = run_keye_inference_full(model, processor, img_path, cnt_prompt, max_tokens=20)
            digit_match = re.search(r'\d+', cnt_res)
            claimed_count = int(digit_match.group()) if digit_match else 0

            # --- 步骤 2: 独立检测 ---
            det_prompt = f"Detect <{target}> and provide bounding boxes in [xmin, ymin, xmax, ymax]. If none, return 'none'."
            det_res, det_confs = run_keye_inference_full(model, processor, img_path, det_prompt)
            coords, initial_rej = parse_keye_nsr_output(det_res)

            # --- 步骤 3: 仲裁逻辑 (Consistency Arbitration) ---
            final_res, final_confs, final_rej = det_res, det_confs, initial_rej
            
            # 如果 计数任务说有(>0) 但 检测任务说没有，或者 计数说没(0) 但检测画了框
            if (claimed_count > 0 and len(coords) == 0) or (claimed_count == 0 and len(coords) > 0):
                stats["arb_triggered"] += 1
                arb_prompt = (f"Self-Correction Task: Previously you mentioned there are {claimed_count} <{target}>, "
                              f"but your detection output was '{det_res}'. Re-examine the image: are there really any <{target}>? "
                              f" provide boxes in [xmin, ymin, xmax, ymax] or 'none'.")
                final_res, final_confs = run_keye_inference_full(model, processor, img_path, arb_prompt)
                _, final_rej = parse_keye_nsr_output(final_res)

            # --- 最终评分 ---
            stats["count"] += 1
            if final_rej:
                stats["hard_rej"] += 1
                score = 1.0
            else:
                # CP-NSR 公式：对每一个幻觉框应用 (1-conf) 惩罚
                score = 1.0
                if len(final_confs) > 0:
                    for c_k in final_confs:
                        score *= (1.0 - min(c_k, 0.95)) # 避免极端分值
                else:
                    # 输出了文字但没坐标也没拒识
                    score = 0.4 
            
            stats["weighted_score"] += score

        if stats["count"] > 0:
            all_results[dim_name] = {
                "count": stats["count"],
                "hard_acc": stats["hard_rej"] / stats["count"],
                "arb_rate": stats["arb_triggered"] / stats["count"],
                "nsr_score": stats["weighted_score"] / stats["count"]
            }

    # --- 汇总报告 ---
    print("\n" + "="*100)
    print(f"{'Dimension':<25} | {'Count':<8} | {'Arb Rate':<10} | {'Hard Rej Acc':<15} | {'NSR Score'}")
    print("-" * 100)
    for dim, res in all_results.items():
        print(f"{dim:<25} | {res['count']:<8} | {res['arb_rate']:9.2%} | {res['hard_acc']:14.2%} | {res['nsr_score']:.4f}")
    print("="*100)

if __name__ == "__main__":
    main()
