import argparse
import os
import json
import torch
import numpy as np
import re
from tqdm import tqdm
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
from qwen_vl_utils import process_vision_info

# --- 1. 核心评估工具 ---

def get_box_confidence(outputs, tokenizer, input_len):
    """
    提取生成序列中 Bbox 坐标 Token 的平均概率
    outputs: model.generate 的返回对象 (含 scores)
    input_len: 输入 prompt 的 token 长度
    """
    # 转换 logits 为概率
    gen_ids = outputs.sequences[:, input_len:]
    transition_scores = outputs.scores  
    
    digit_probs = []
    for i in range(len(transition_scores)):
        token_id = gen_ids[0, i]
        token_text = tokenizer.decode([token_id]).strip()
        
        # 将 logit 转为 softmax 概率
        probs = torch.softmax(transition_scores[i][0], dim=-1)
        token_prob = probs[token_id].item()
        
        # 匹配数字 token
        if re.search(r'\d+', token_text):
            digit_probs.append(token_prob)
            
    # 每 4 个数字构成一个框 [xmin, ymin, xmax, ymax]
    box_confs = []
    for j in range(0, len(digit_probs) // 4 * 4, 4):
        avg_conf = np.mean(digit_probs[j : j+4])
        box_confs.append(avg_conf)
    return box_confs

def parse_output(output_text):
    """解析输出：返回 (是否语义拒识, 坐标框列表)"""
    text_lower = output_text.lower()
    negative_words = ["none", "no object", "not found", "isn't", "unable", "cannot see", "未找到"]
    is_rejected = any(word in text_lower for word in negative_words)
    
    # 正则：匹配 [x1, y1, x2, y2]
    coords = re.findall(r"\[([\d\.]+),\s*([\d\.]+),\s*([\d\.]+),\s*([\d\.]+)\]", output_text)
    return is_rejected, coords

# --- 2. 仲裁推理驱动 ---

def run_inference(model, processor, image, prompt):
    """单样本推理，返回 (回复文本, 原始输出对象, 输入长度)"""
    messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, return_tensors="pt").to(model.device)
    
    input_len = inputs.input_ids.shape[1]
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=128, 
            do_sample=False,
            output_scores=True,
            return_dict_in_generate=True
        )
    
    gen_ids = outputs.sequences[:, input_len:]
    response = processor.batch_decode(gen_ids, skip_special_tokens=True)[0]
    return response, outputs, input_len

# --- 3. 主评测流程 ---

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", type=str, default='/root/autodl-tmp/LLaVA-OneVision-1.5/test/logo_test_labels.json')
    parser.add_argument("--img_dir", type=str, default='/root/autodl-tmp/LLaVA-OneVision-1.5/test/logo')
    parser.add_argument("--model_path", type=str, default="LLaVA-OneVision-1.5-8B-Instruct")
    args = parser.parse_args()

    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype="auto", device_map="auto", trust_remote_code=True).eval()
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)

    with open(args.json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 负样本评测
    neg_metrics = {"count": 0, "hard_rej": 0, "weighted_score": 0.0}

    for item in tqdm(data, desc="Evaluating Negative Samples"):
        if "negative" not in item: continue
        
        img_p = os.path.join(args.img_dir, item["image"])
        if not os.path.exists(img_p): continue
        img = Image.open(img_p).convert("RGB")
        target = item["negative"]

        # 阶段 1: 计数与检测初测
        count_res, _, _ = run_inference(model, processor, img, f"How many '{target}' are in this image? Answer with a number only.")
        det_res, det_outputs, det_in_len = run_inference(model, processor, img, f"Detect '{target}'. If not found, return 'none'.")

        # 一致性分析
        c_match = re.search(r'\d+', count_res)
        exp_count = int(c_match.group()) if c_match else 0
        is_rej, coords = parse_output(det_res)

        # 阶段 2: 逻辑仲裁 (如果计数不为0但检测说没有，或者计数为0但检测画了框)
        if (exp_count > 0 and is_rej) or (exp_count == 0 and len(coords) > 0):
            arb_prompt = f"Review: I see a discrepancy. You said count is {exp_count} but detection shows {len(coords)} boxes for '{target}'. Re-scan carefully and output boxes or 'none'."
            det_res, det_outputs, det_in_len = run_inference(model, processor, img, arb_prompt)
            is_rej, coords = parse_output(det_res)

        neg_metrics["count"] += 1
        
        # A. 绝对指标：语义拒识且没有框
        if is_rej and len(coords) == 0:
            neg_metrics["hard_rej"] += 1
            neg_metrics["weighted_score"] += 1.0
        
        # B. 相对指标：存在幻觉框
        elif len(coords) > 0:
            box_confs = get_box_confidence(det_outputs, processor.tokenizer, det_in_len)
            if box_confs:
                # 连乘惩罚公式：Score = Π (1 - c_k)
                sample_score = 1.0
                for c_k in box_confs:
                    sample_score *= (1.0 - c_k)
                neg_metrics["weighted_score"] += sample_score
            else:
                neg_metrics["weighted_score"] += 0.0 # 有框但提取不出概率，视为极度自信的幻觉
        
        # C. 既不拒绝也不画框的样本
        else:
            neg_metrics["weighted_score"] += 0.5

    # 输出结果
    cnt = neg_metrics["count"]
    if cnt > 0:
        print("\n" + "="*50)
        print(f"Negative Sample Evaluation Report")
        print("-" * 50)
        print(f"Total Samples:  {cnt}")
        print(f"Hard Rej Rate:  {neg_metrics['hard_rej']/cnt:.2%}")
        print(f"Weighted Score: {neg_metrics['weighted_score']/cnt:.4f}")
        print("="*50)

if __name__ == "__main__":
    main()
