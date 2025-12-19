import argparse
import torch
from PIL import Image
import gradio as gr
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
import os

import re
import html

os.environ["NO_PROXY"] = "localhost,127.0.0.1,0.0.0.0"
os.environ["no_proxy"] = os.environ["NO_PROXY"]

TASK_SUFFIX = (
    "Your task: 1. First, Think through the question step by step, enclose your reasoning process "
    "in <think>...</think> tags. 2. Then provide the correct answer inside <answer>...</answer> tags. "
    "3. No extra information or text outside of these tags."
)

MODEL = None
PROCESSOR = None


def parse_think_answer(text: str):
    if not text:
        return "", "", ""

    think = ""
    answer = ""

    m_think = re.search(r"<think>(.*?)</think>", text, flags=re.S)
    if m_think:
        think = m_think.group(1).strip()

    m_ans = re.search(r"<answer>(.*?)</answer>", text, flags=re.S)
    if m_ans:
        answer = m_ans.group(1).strip()

    if not answer:
        answer = text.strip()

    return think, answer, text.strip()


def render_answer_card(answer: str):
    safe = html.escape(answer)
    return f"""
    <div style="
        padding:16px 18px;
        border-radius:14px;
        border:1px solid rgba(0,0,0,0.08);
        box-shadow:0 6px 18px rgba(0,0,0,0.06);
        background: linear-gradient(180deg, #ffffff 0%, #fafafa 100%);
        font-size:18px;
        line-height:1.5;
    ">
      <div style="font-size:12px; opacity:0.65; margin-bottom:6px;">Final Answer</div>
      <div style="font-weight:700;">{safe}</div>
    </div>
    """


def load_model(model_path: str, use_flash_attn: bool = True):
    global MODEL, PROCESSOR

    kwargs = dict(
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    if use_flash_attn:
        kwargs["attn_implementation"] = "flash_attention_2"

    try:
        MODEL = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, **kwargs)
    except Exception as e:
        if use_flash_attn:
            print(f"[Warn] flash_attention_2 load failed, fallback to default attention. Error: {e}")
            kwargs.pop("attn_implementation", None)
            MODEL = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, **kwargs)
        else:
            raise

    MODEL.eval()
    PROCESSOR = AutoProcessor.from_pretrained(model_path)

    print(f"[Info] Model loaded successfully!")
    print("=== Generation defaults ===")
    print(MODEL.generation_config)
    print("===========================")


@torch.inference_mode()
def infer(image: Image.Image, user_query: str):
    if MODEL is None or PROCESSOR is None:
        return "[Error] Model not loaded."

    if image is None:
        return "[Error] Please upload an image."
    if user_query is None or not user_query.strip():
        return "[Error] Question is empty."

    image = image.convert("RGB")
    final_question = f"{user_query.strip()} {TASK_SUFFIX}"

    # the format of multimodal chatbot
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": final_question},
            ],
        }
    ]

    text_prompt = PROCESSOR.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = PROCESSOR(
        text=[text_prompt],
        images=[image],
        padding=True,
        return_tensors="pt",
    ).to(MODEL.device)

    generated_ids = MODEL.generate(
        **inputs,
        max_new_tokens=1024,
        do_sample=False,
    )

    generated_ids_trimmed = generated_ids[:, inputs.input_ids.shape[1]:]
    prediction = PROCESSOR.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True
    )[0]

    think, answer, raw = parse_think_answer(prediction.strip())
    if not think:
        think = "no think processed ..."

    return render_answer_card(answer), think


def build_demo():
    with gr.Blocks(title="ColonR1 Gradio Demo") as demo:
        gr.Markdown("## ColonR1 Web Demo")
        gr.Markdown("Usage: Please upload a colonoscopy image, ask a question. Thinking process is shown below the final answer. **Recommend:** Click the bottom provided examples for a quick test. Read our paper (https://arxiv.org/abs/2512.03667) and project page (https://github.com/ai4colonoscopy/Colon-X).")

        with gr.Row():
            img_in = gr.Image(type="pil", label="Upload Image")

            with gr.Column():
                q_in = gr.Textbox(
                    label="Expert Question",
                    placeholder="Type your question here...",
                    lines=3,
                )

                run_btn = gr.Button("Run", variant="primary")

                answer_html = gr.HTML(label="Answer")

                with gr.Accordion("Thinking...", open=True):
                    think_tb = gr.Textbox(lines=6, label="Raw")

        run_btn.click(
            fn=infer,
            inputs=[img_in, q_in],
            outputs=[answer_html, think_tb],
        )

        gr.Examples(
            examples=[
                # example 2
                ["./test_examples/02/102.jpg", 
                 "What intervention phase does this image represent? <A> resection margin, <B> dyed resection margin, <C> resected polyp, <D> dyed lifted polyp"],
                # example 4
                ["./test_examples/04/adenoma_02_1.jpg", 
                 "Which imaging technique does this image use? <A> Linked Color Imaging (LCI), <B> Narrow Band Imaging (NBI), <C> Blue Light Imaging (BLI), <D> Flexible Imaging Color Enhancement (FICE), <E> White Light Imaging (WLI)"]
            ],
            inputs=[img_in, q_in],
            fn=infer,
            run_on_click=True,
            label="Examples (ps. click to auto load & run)",
        )

    return demo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, help="Path to model checkpoint")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="Enable gradio share link")
    parser.add_argument("--no_flash_attn", action="store_true", help="Disable flash_attention_2")

    args = parser.parse_args()

    print(f"[Info] Loading model from {args.model_path} ...")
    load_model(args.model_path, use_flash_attn=not args.no_flash_attn)

    demo = build_demo()
    demo.queue(max_size=32).launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()