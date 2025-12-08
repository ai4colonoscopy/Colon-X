import argparse
import os
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

def main(args):
    if not os.path.exists(args.image_path):
        print(f"[Error] Image not found: {args.image_path}")
        return

    print(f"[Info] Loading model from {args.model_path}...")
    try:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto"
        )
        model.eval()
        processor = AutoProcessor.from_pretrained(args.model_path)
        print("[Info] Model loaded successfully!")
    except Exception as e:
        print(f"[Error] Failed to load model: {e}")
        return

    TASK_SUFFIX = "Your task: 1. First, Think through the question step by step, enclose your reasoning process in <think>...</think> tags. 2. Then provide the correct answer inside <answer>...</answer> tags. 3. No extra information or text outside of these tags."

    image_check = Image.open(args.image_path).convert("RGB")

    while True:
        user_query = input("[Input] Please enter your question: ").strip()
        
        if user_query.lower() in ['exit', 'quit']:
            print("Exiting...")
            break
        
        if not user_query:
            print("[Warning] Question is empty, skipping.")
            continue
        
        final_question = f"{user_query}{TASK_SUFFIX}"
        
        image = Image.open(args.image_path).convert("RGB")
        
        message = [
            {
                "role": "user", 
                "content": [
                    {"type": "image", "image": args.image_path}, 
                    {"type": "text", "text": question}
                ]
            }
        ]

        text_prompt = processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
        
        inputs = processor(
            text=[text_prompt], 
            images=[image], 
            padding=True, 
            return_tensors="pt"
        ).to(model.device)
        
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs, 
                max_new_tokens=1024, 
                do_sample=False
            )
        
        generated_ids_trimmed = generated_ids[:, inputs.input_ids.shape[1]:]
        prediction = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0]
        
        print(prediction.strip())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, help="Path to the model checkpoint")
    parser.add_argument("--image_path", type=str, help="Path to the single image file")
    
    args = parser.parse_args()
    main(args)