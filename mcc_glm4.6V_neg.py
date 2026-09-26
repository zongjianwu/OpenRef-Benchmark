import argparse
import os
import json
import torch
import numpy as np
import re
from tqdm import tqdm
from PIL import Image
from transformers import AutoProcessor, Glm4vForConditionalGeneration

# 开启性能加速
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

# --- 1. 核心工具函数 ---

def get_glm_box_confidences(outputs, generated_ids, prompt_len, processor):
    """ 提取生成的坐标数字 Token 的平均置信度 """
    if not hasattr(outputs, "scores"): return []
    logits = outputs.scores
    probs = [torch.softmax(l.float(), dim=-1) for l in logits]
    batch_size = generated_ids.shape[0]
    batch_box_confs = []

    for b in range(batch_size):
        sample_gen_ids = generated_ids[b, prompt_len:]
        token_probs = []
        for i, tid in enumerate(sample_gen_ids):
            if i < len(probs):
                t_text = processor.decode([tid]).strip()
                if t_text.isdigit():
                    token_probs.append(probs[i][b, tid].item())
        
        box_confs = [np.mean(token_probs[j:j+4]) for j in range(0, len(token_probs)//4*4, 4)]
        batch_box_confs.append(box_confs)
    return batch_box_confs

def parse_negative_output(text):
    """ 解析输出：返回 (坐标列表, 是否语义拒识) """
    text_lower = text.lower()
    neg_words = ["none", "not found", "isn't", "no ", "cannot see", "unable", "zero", "not exist", "[]"]
    semantic_rejected = any(word in text_lower for word in neg_words)
    coords = re.findall(r"(\d+),\s*(\d+),\s*(\d+),\s*(\d+)", text)
    return coords, semantic_rejected

# --- 2. 推理执行引擎 ---

def run_glm_inference(model, processor, image, prompt):
    """ 封装单次推理过程 """
    messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
    inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, 
                                          return_dict=True, padding=True, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        outputs = model.generate(**inputs, max_new_tokens=128, do_sample=False, 
                                 output_scores=True, return_dict_in_generate=True)
    
    gen_ids = outputs.sequences
    p_len = inputs["input_ids"].shape[1]
    response = processor.batch_decode(gen_ids[:, p_len:], skip_special_tokens=True)[0]
    return response, outputs, p_len

def evaluate_sample_with_arbitration(model, processor, item, img_dir):
    img_path = os.path.join(img_dir, item["image"])
    raw_img = Image.open(img_path).convert("RGB")
    target = item['negative']

    # Step A: 独立计数任务
    cnt_q = f"How many <ref>{target}</ref> are there in the image? Answer with a single integer."
    cnt_res, _, _ = run_mcp_inference = run_glm_inference(model, processor, raw_img, cnt_q)
    digit_match = re.search(r'\d+', cnt_res)
    claimed_count = int(digit_match.group()) if digit_match else 0

    # Step B: 独立检测任务
    det_q = f"Detect <ref>{target}</ref>. If it exists, return [[xmin, ymin, xmax, ymax]]. If not exist, return 'none'."
    det_res, det_out, det_plen = run_glm_inference(model, processor, raw_img, det_q)
    coords, is_rej = parse_negative_output(det_res)

    # Step C: 仲裁 (Arbitration)
    final_res, final_out, final_plen = det_res, det_out, det_plen
    arb_triggered = False

    # 逻辑矛盾判断：计数>0但没框，或计数=0但画了框
    if (claimed_count > 0 and len(coords) == 0) or (claimed_count == 0 and len(coords) > 0):
        arb_triggered = True
        arb_q = (f"You said {claimed_count} <ref>{target}</ref> and detection bboxes was '{det_res}'."
                 f"Re-examine carefully: If exists, return [[xmin, ymin, xmax, ymax]]. If not, return 'none'.")
        final_res, final_out, final_plen = run_glm_inference(model, processor, raw_img, arb_q)

    # 最终结果解析
    final_coords, final_is_rej = parse_negative_output(final_res)
    box_confs = get_glm_box_confidences(final_out, final_out.sequences, final_plen, processor)[0]
    
    # 评分逻辑
    is_hard_success = (len(final_coords) == 0 and final_is_rej)
    score = 1.0
    if not is_hard_success:
        if len(final_coords) > 0:
            # 幻觉惩罚：CP-NSR 公式
            if not box_confs: score = 0.0
            else:
                for c in box_confs: score *= (1.0 - min(c, 0.98))
        else:
            score = 0.5 # 语义模糊

    return score, is_hard_success, arb_triggered

# --- 4. 主评测流程 ---

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_root", type=str, default="test2")
    parser.add_argument("--model_path", type=str, default="/root/autodl-tmp/local_weights/ZhipuAI/GLM-4.6V-Flash")
    args = parser.parse_args()

    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    model = Glm4vForConditionalGeneration.from_pretrained(args.model_path, torch_dtype=torch.bfloat16, 
                                                         device_map="auto", trust_remote_code=True).eval()

    subsets = sorted([d for d in os.listdir(args.test_root) if os.path.isdir(os.path.join(args.test_root, d))])
    all_metrics = []

    print("\n" + "="*105)
    print(f"{'Subset Name':<25} | {'Count':<6} | {'Arb Rate':<10} | {'Hard Rej↑':<12} | {'CP-NSR↑'}")
    print("-" * 105)

    for dim in subsets:
        json_path = os.path.join(args.test_root, f"{dim}_test_labels.json")
        img_dir = os.path.join(args.test_root, dim)
        if not os.path.exists(json_path): continue

        with open(json_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        tasks = [item for item in dataset if item.get("negative")]
        if not tasks: continue

        dim_stats = {"count": 0, "hard": 0, "score": 0.0, "arb": 0}

        for item in tqdm(tasks, desc=f"Eval {dim}", leave=False):
            s, h, a = evaluate_sample_with_arbitration(model, processor, item, img_dir)
            dim_stats["count"] += 1
            dim_stats["hard"] += (1 if h else 0)
            dim_stats["score"] += s
            dim_stats["arb"] += (1 if a else 0)

        if dim_stats["count"] > 0:
            m = {
                "name": dim, "count": dim_stats["count"],
                "arb": dim_stats["arb"]/dim_stats["count"],
                "hard": dim_stats["hard"]/dim_stats["count"],
                "nsr": dim_stats["score"]/dim_stats["count"]
            }
            all_metrics.append(m)
            print(f"{m['name']:<25} | {m['count']:<6} | {m['arb']:10.2%} | {m['hard']:12.2%} | {m['nsr']:.4f}")

    if all_metrics:
        print("-" * 105)
        avg_hard = np.mean([m['hard'] for m in all_metrics])
        avg_nsr = np.mean([m['nsr'] for m in all_metrics])
        print(f"{'MEAN (All Dimensions)':<25} | {'-':<6} | {'-':<10} | {avg_hard:12.2%} | {avg_nsr:.4f}")
    print("="*105)

if __name__ == "__main__":
    main()
