import argparse
import os
import json
from PIL import Image
from tqdm import tqdm
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

def get_processed_keys(data):
    processed = set()
    for item in data:
        image_id = item.get("image")
        human_val = ""
        for conv in item.get("conversations", []):
            if conv.get("from") == "human":
                human_val = conv.get("value", "")
                break
        
        if image_id and human_val:
            processed.add((image_id, human_val))
    return processed

def main(args):
    with open(args.json_file, "r", encoding="utf-8") as f:
        all_data = json.load(f)

    existing_data = []
    if os.path.exists(args.output_path):
        print(f"[Info] Loading existing results from {args.output_path}...")
        with open(args.output_path, "r", encoding="utf-8") as f:
            existing_data = json.load(f)
    
    processed_keys = get_processed_keys(existing_data)
    if processed_keys:
        print(f"[Info] Found {len(processed_keys)} processed items.")

    data_to_process = []
    for item in all_data:
        image_id = item.get("image")
        human_val = ""
        for conv in item.get("conversations", []):
            if conv.get("from") == "human":
                human_val = conv.get("value", "")
                break
        
        if (image_id, human_val) not in processed_keys:
            data_to_process.append(item)

    print(f"[Info] Items remaining to process: {len(data_to_process)}")

    if not data_to_process:
        print("All data processed. Exiting.")
        return

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto"
    )
    model.eval()

    processor = AutoProcessor.from_pretrained(args.model_path)

    TASK_SUFFIX = "Your task: 1. First, Think through the question step by step, enclose your reasoning process in <think>...</think> tags. 2. Then provide the correct answer inside <answer>...</answer> tags. 3. No extra information or text outside of these tags."

    if os.path.exists(args.output_path) and len(existing_data) > 0:
        f = open(args.output_path, "r+", encoding="utf-8")
        f.seek(0, os.SEEK_END)
        pos = f.tell() - 1
        while pos > 0:
            f.seek(pos)
            char = f.read(1)
            if char == ']':
                f.seek(pos)
                f.truncate()
                break
            pos -= 1
        needs_comma = True
    else:
        f = open(args.output_path, "w", encoding="utf-8")
        f.write("[\n")
        needs_comma = False

    try:
        for i, item in enumerate(tqdm(data_to_process, desc="Inference")):

            raw_question = ""
            for conv in item.get("conversations", []):
                if conv.get("from") == "human":
                    raw_question = conv.get("value", "").replace("<image>\n", "").strip()
                    break
            
            if not raw_question:
                continue

            question = f"{raw_question}{TASK_SUFFIX}"
            full_image_path = os.path.join(args.image_dir, item["image"])
            
            if not os.path.exists(full_image_path):
                print(f"[Error] Image missing: {full_image_path}")
                continue

            image = Image.open(full_image_path).convert("RGB")
            message = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}]
            
            text = processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)
            
            with torch.no_grad():
                generated_ids = model.generate(**inputs, max_new_tokens=1024, do_sample=False)
            
            generated_ids_trimmed = generated_ids[:, inputs.input_ids.shape[1]:]
            prediction = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0]
            
            item["conversations"].append({
                "from": "prediction",
                "value": prediction.strip()
            })
            
            if needs_comma:
                f.write(",\n")
            
            json.dump(item, f, indent=4, ensure_ascii=False)
            
            f.flush() 
            needs_comma = True
        f.write("\n]")

    except Exception as e:
        print(f"\n[Error] Inference interrupted: {e}")
        print("[Info] Data saved up to the crash point. Please manually check file ending (add ']') before resuming.")
    finally:
        f.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--json_file", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()
    main(args)