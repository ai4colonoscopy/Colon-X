import os
import json
import base64
import time
import io
import hashlib
import argparse
from openai import OpenAI
from tqdm import tqdm
from PIL import Image

key = 'xxxxxxxxxxxxxx' # Please fill in the information as needed.
client = OpenAI(
    base_url="xxxxxxxxxxxxxx", # Please fill in the information as needed.
    api_key=key,
    timeout=120
)

def get_unique_task_id(item):
    image_id = item.get('image', item.get('id', '')).strip()
    question = next(
        (conv['value'] for conv in item.get('conversations', []) if conv.get('from') == 'human'), 
        ''
    ).strip()
    
    if not image_id or not question:
        print(f"Warning: Missing 'id' or 'question' for an item. Item: {item}")
        return None

    unique_string = f"{image_id}::{question}"
    return hashlib.sha256(unique_string.encode('utf-8')).hexdigest()


def encode_image(image_path):
    with Image.open(image_path) as img:
        if img.mode != 'RGB':
            img = img.convert('RGB')
        output_buffer = io.BytesIO()
        img.save(output_buffer, format="JPEG") 
        byte_data = output_buffer.getvalue()
        return base64.b64encode(byte_data).decode('utf-8')


def main(args):
    processed_task_hashes = set()
    if os.path.exists(args.output_path) and os.path.getsize(args.output_path) > 0:
        print(f"Inspecting existing output file: {args.output_path}")
        try:
            with open(args.output_path, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
            
            for item in existing_data:
                task_hash = get_unique_task_id(item)
                if task_hash:
                    processed_task_hashes.add(task_hash)
        except json.JSONDecodeError:
            print(f"Error: Output file at {args.output_path} appears to be corrupted.")
            print("Please fix it (e.g., ensure it's a valid JSON array) and rerun.")
            return
    
    print(f"Found {len(processed_task_hashes)} unique processed tasks.")

    with open(args.json_file, "r", encoding='utf-8') as f:
        all_data = json.load(f)

    items_to_process = [
        item for item in all_data 
        if get_unique_task_id(item) not in processed_task_hashes
    ]
    
    if not items_to_process:
        print("All tasks have been processed. Nothing new to predict.")
        return

    print(f"Found {len(items_to_process)} new tasks to process.")


    if processed_task_hashes:
        with open(args.output_path, 'rb+') as f:
            f.seek(0, os.SEEK_END)
            if f.tell() > 0:
                f.seek(-1, os.SEEK_END)
                if f.read(1) == b']':
                    f.seek(-1, os.SEEK_END)
                    f.truncate()
    else:
        with open(args.output_path, 'w', encoding='utf-8') as f:
            f.write('[')

    with open(args.output_path, 'a', encoding='utf-8') as f_out:
        with tqdm(total=len(items_to_process)) as pbar:
            for i, item in enumerate(items_to_process):
                if processed_task_hashes or i > 0:
                    f_out.write(',')

                image_filename = item.get('image', item.get('id'))
                image_path = os.path.join(args.image_dir, image_filename)

                base64_image = encode_image(image_path)
                if base64_image is None:
                    pbar.update(1)
                    continue

                question_text = item["conversations"][-1]["value"]

                messages = [
                    {"role": "system", "content": "As a medical professional, analyze the following clinical image from a colonoscopy for a medical research study. The image is for research purposes only."},
                    {"role": "user", "content": [
                        {"type": "text", "text": question_text},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]}
                ]
                
                try:
                    response = client.chat.completions.create(
                        model=args.model, 
                        messages=messages, 
                        temperature=0.5,
                        max_tokens=args.max_tokens
                    )

                    prediction = "" 
                    if response.choices and response.choices[0].message:
                        prediction = response.choices[0].message.content or ""
                    if not prediction:
                        print(f"[Warning] '{image_filename}' get enmpty prediction, printing full API response for debugging:")
                        print(response.model_dump_json(indent=2))

                except Exception as e:
                    print(f"API request failed for task with image '{image_filename}': {e}. Skipping.")
                    pbar.update(1)
                    continue

                item["conversations"].append({"from": "prediction", "value": prediction})
                
                json.dump(item, f_out, indent=4, ensure_ascii=False)
                f_out.flush()

                pbar.update(1)
                time.sleep(1)

    with open(args.output_path, 'a', encoding='utf-8') as f_out:
        f_out.write(']')
    
    print("All done !!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process medical images with questions and handle interruptions gracefully.")
    parser.add_argument("--model", type=str, help="Name of the model to use.")
    parser.add_argument("--image_dir", type=str, help="Directory containing the images.")
    parser.add_argument("--json_file", type=str, help="Path to the input JSON file with tasks.")
    parser.add_argument("--output_path", type=str, help="Path to the output JSON file.")
    parser.add_argument("--max_tokens", type=int, default=4096, help="Maximum number of tokens for the response.")
    args = parser.parse_args()

    main(args)

