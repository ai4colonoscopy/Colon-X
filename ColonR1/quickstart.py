import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from PIL import Image
import warnings
import os

warnings.filterwarnings('ignore')
device = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_PATH = "ai4colonoscopy/ColonR1"
IMAGE_PATH = "ColonR1/serve/test_examples/02/102.jpg"
Question = "Does the image contain a polyp? Answer me with Yes or No."

print(f"[Info] Loading model from {MODEL_PATH}...")
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto"
)
model.eval()

processor = AutoProcessor.from_pretrained(MODEL_PATH)

if not os.path.exists(IMAGE_PATH):
    raise FileNotFoundError(f"Image not found at {IMAGE_PATH}. Please provide a valid image path.")

image = Image.open(IMAGE_PATH).convert("RGB")

TASK_SUFFIX = (
    "Your task: 1. First, Think through the question step by step, enclose your reasoning process "
    "in <think>...</think> tags. 2. Then provide the correct answer inside <answer>...</answer> tags. "
    "3. No extra information or text outside of these tags."
)

final_question = f"{Question}\n{TASK_SUFFIX}"

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": IMAGE_PATH},
            {"type": "text", "text": final_question},
        ],
    }
]

print("[Info] Processing inputs...")
text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

inputs = processor(
    text=[text_prompt],
    images=[image],
    padding=True,
    return_tensors="pt",
).to(device)


print("[Info] Generating response...")
with torch.no_grad():
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=1024,
        do_sample=False
    )

generated_ids_trimmed = generated_ids[:, inputs.input_ids.shape[1]:]
output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0]

print(output_text)
