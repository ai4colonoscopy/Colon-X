import argparse
import json
import os
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
from tqdm import tqdm


def main(args):

    model = AutoModelForImageTextToText.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(args.model_path)

    print(f"Loading data from: {args.json_file}")

    with open(args.json_file, 'r', encoding='utf-8') as f:
        input_data = json.load(f)

    results = []
    done_ids = set()
    if os.path.exists(args.output_path):
        with open(args.output_path, 'r', encoding='utf-8') as f:
            existing_results = json.load(f)
        
        if isinstance(existing_results, list):
            results = existing_results
            done_ids = {item.get("image") for item in results if item.get("image")}
        else:
            print(f"Existing output file is not a list. Starting from scratch.")
    else:
        print("No existing output file found. Starting a new inference task.")

    progress_bar = tqdm(input_data, initial=len(results), total=len(input_data))
    for item in progress_bar:
        relative_image_path = item.get("image")

        if relative_image_path in done_ids:
            continue

        full_image_path = os.path.join(args.image_dir, relative_image_path)

        human_prompt = ""
        for conv in item["conversations"]:
            if conv.get("from") == "human":
                human_prompt = conv.get("value", "").replace("<image>\n", "").strip()
                break

        image = Image.open(full_image_path)
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": human_prompt},
                    {"type": "image", "image": image}
                ]
            }
        ]

        inputs = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt"
        ).to(model.device)

        input_len = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            generation = model.generate(**inputs, max_new_tokens=200, do_sample=False)
            generation = generation[0][input_len:]
            decoded_response = processor.decode(generation, skip_special_tokens=True)

        item["conversations"].append({
            "from": "prediction",
            "value": decoded_response.strip()
        })
        results.append(item)


    print(f"Saving results to {args.output_path}...")
    with open(args.output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    print("All done!")

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Batch inference script for MedGemma")
    parser.add_argument("--model_path", type=str, help="Path or Hugging Face ID of the model.")
    parser.add_argument("--json_file", type=str, help="Path to the input JSON file.")
    parser.add_argument("--image_dir", type=str, help="The base directory where images are stored.")
    parser.add_argument("--output_path", type=str, help="Path to save the output JSON file with predictions.")
    args = parser.parse_args()
    main(args)
