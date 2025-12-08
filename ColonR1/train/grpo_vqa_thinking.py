import os
import re
import json
import math
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional, Any, List

from datasets import load_dataset, load_from_disk
from transformers import Qwen2VLForConditionalGeneration

from math_verify import parse, verify
from trainer.grpo_trainer import Qwen2VLGRPOTrainer
from trl import GRPOConfig, GRPOTrainer, ModelConfig, ScriptArguments, TrlParser, get_peft_config
import torch

from sentence_transformers import SentenceTransformer, util
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl


class AdaptiveKLCallback(TrainerCallback):
    '''
    Dynamically adjust the KL divergence coefficient
    Cosine annealing strategy to gradually reduce the beta value
    '''
    def __init__(self, initial_beta, final_beta):
        self.initial_beta = initial_beta
        self.final_beta = final_beta
        print(f"Adaptive KL divergence is enabled: beta will decrease from {initial_beta} to {final_beta}")

    def on_step_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        trainer = kwargs.get("trainer")
        if trainer is not None and hasattr(trainer, "beta"):
            progress = state.global_step / state.max_steps
            cosine_progress = (1 + math.cos(math.pi * progress)) / 2
            new_beta = self.final_beta + cosine_progress * (self.initial_beta - self.final_beta)
            trainer.beta = new_beta
            if state.global_step > 0 and state.global_step % 100 == 0:
                trainer.log({"custom_beta": new_beta})


@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.
    """
    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format", "length"],
        metadata={"help": "List of reward functions. Possible values: 'accuracy', 'format', 'length'"}
    )
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image"},
    )
    initial_beta: Optional[float] = field(
        default=0.6,
        metadata={"help": "The initial beta value for KL annealing."},
    )
    final_beta: Optional[float] = field(
        default=0.1,
        metadata={"help": "The final beta value for KL annealing."},
    )
    bad_case_log_file: Optional[str] = field(
        default=None, 
        metadata={"help": "Path to log bad cases (reward < 0.5) as a JSON file."}
    )
    image_root_dir: Optional[str] = field(
        default="",
        metadata={"help": "Base directory to prepend to relative image paths in the dataset."}
    )


# ------------------------ Our reward function ------------------------ #
SBERT_LOCAL_PATH = 'cache/download-weights/all-MiniLM-L6-v2' # SBERT model
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
sbert_model = SentenceTransformer(SBERT_LOCAL_PATH, device=DEVICE)
print(f"Sentence-BERT load from '{SBERT_LOCAL_PATH}' to {DEVICE}")

CATEGORIES = [
    "polyp", "adenoma", "hyperplastic lesion", "high grade dysplasia", "high grade adenoma",
    "low grade adenoma", "sessile serrated lesion", "traditional serrated adenoma", "adenocarcinoma",
    "colorectal cancer", "invasive carcinoma", "suspicious precancerous lesion", "tumor",
    "erosion", "ulcer", "ulcerative colitis", "ulcerative colitis grade 0-1",
    "ulcerative colitis grade 1-2", "ulcerative colitis grade 2-3", "ulcerative colitis grade 0",
    "ulcerative colitis grade 1", "ulcerative colitis grade 2", "ulcerative colitis grade 3",
    "aphthae", "inflammatory bowel disease", "chylous-cysts", "inflammatory", "blood fresh",
    "bleeding", "blood hematin", "blood in lumen", "angiectasia", "vascular anomalies",
    "vascular lesions", "colon diverticula", "erythema", "hemorrhoid", "lymphangiectasia",
    "lymphangiectasias-nodular", "stenoses", "villous-oedemas"
]

def extract_answer_content(text):
    '''
    Extract the answer content of the reference and model prediction
    '''
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return text.strip()

def parse_answer(text):
    '''
    Extracting answers to multiple-choice questions
    '''
    match = re.match(r"<([A-Z])>\s*(.*?)(?:\s*\((.*)\))?\s*$", text, re.IGNORECASE | re.DOTALL)
    if not match:
        return None
    letter, content, abbr = match.groups()
    letter = letter.strip().lower()
    content = content.strip().lower()
    abbr = (abbr or "").strip().lower()
    return letter, content, abbr

def is_mcq_task(gt):
    '''
    Whether a question is a MCQ can be determined by whether the GT can be parsed by parse_answer().
    '''
    return parse_answer(gt) is not None

def is_yes_no_task(gt):
    '''
    Determine if it is a yes-or-no question.
    '''
    return gt.strip().lower() in ["yes", "no"]


def accuracy_reward(completions, **kwargs):
    '''
    Accuracy reward
    Scores are automatically calculated based on question type.
    1. yes-or-no question: 1.0 for exact match, 0 for no match.
    2. multiple choice question: 2.0 for exact match, 1.0 for partial match (letter or content match), no points otherwise.
    3. open question: SBERT is used to calculate the cosine similarity between the prediction and the reference as the score.
    '''
    prompts = kwargs.get("prompts")
    solution = kwargs.get("solution")
    problem_list = kwargs.get("problem") 

    num_completions = len(completions)
    if num_completions == 0:
        return []

    num_generations_per_prompt = kwargs.get("num_generations_per_prompt", 0)

    if num_generations_per_prompt > 0:
        num_prompts = num_completions // num_generations_per_prompt
    else:
        raise ValueError(
            "[Error] 'num_generations_per_prompt' is missing! "
            "Please check your Trainer configuration."
        )

    rewards = []
    openset_preds, openset_gts, openset_indices = [], [], []

    for i, (completion, sol) in enumerate(zip(completions, solution)):
        pred_ans = extract_answer_content(completion[0]["content"])
        gt_ans = extract_answer_content(sol)

        # Standardize strings, convert to lowercase, remove spaces
        pred_ans_norm = pred_ans.lower().strip()
        gt_ans_norm = gt_ans.lower().strip()

        score = 0.0
        if is_yes_no_task(gt_ans):
            if gt_ans_norm == pred_ans_norm:
                score = 1.0
        elif is_mcq_task(gt_ans):
            parsed_gt = parse_answer(gt_ans)
            if parsed_gt:
                gt_letter, gt_content, gt_abbr = parsed_gt
                letter_pattern = rf"<{gt_letter}>|\({gt_letter}\)|\b{gt_letter}[\s\.\)>\]]|\b{gt_letter}$"
                letter_match = bool(re.search(letter_pattern, pred_ans_norm, re.IGNORECASE))
                content_match = gt_content in pred_ans_norm if gt_content else False
                abbr_match = gt_abbr in pred_ans_norm if gt_abbr else False
                full_match = letter_match and content_match
                if gt_abbr:
                    full_match = full_match and abbr_match
                if full_match:
                    score = 2.0
                elif letter_match or content_match or (gt_abbr and abbr_match):
                    score = 1.0
        else: # open question
            openset_preds.append(pred_ans_norm)
            openset_gts.append(gt_ans_norm)
            openset_indices.append(i)
            rewards.append(-999)
            continue
        rewards.append(score)

    if openset_preds:
        pred_embeddings = sbert_model.encode(openset_preds, convert_to_tensor=True)
        gt_embeddings = sbert_model.encode(openset_gts, convert_to_tensor=True)
        cosine_scores = util.cos_sim(pred_embeddings, gt_embeddings)
        for idx, original_index in enumerate(openset_indices):
             # Ensure index access is safe
            if idx < cosine_scores.shape[0] and idx < cosine_scores.shape[1]:
                 rewards[original_index] = max(0, cosine_scores[idx][idx].item())
            else:
                 print(f"[Warning] Index out of bounds in SBERT scoring: idx={idx}, original_index={original_index}")
                 rewards[original_index] = 0.0 # Assign a default score

    return rewards


def format_reward(completions, **kwargs):
    '''
    Format Rewards
    Force models to conform to the <think></think><answer></answer> format
    '''
    rewards = []
    FORMAT_BONUS = 0.1 
    FORMAT_PENALTY = -0.1

    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"

    for completion in completions:
        content = ""
        if isinstance(completion, (list, tuple)) and len(completion) > 0 and isinstance(completion[0], dict):
            content = completion[0].get("content", "")
        match = re.fullmatch(pattern, content, re.DOTALL)
        if match:
            rewards.append(FORMAT_BONUS)
        else:
            rewards.append(FORMAT_PENALTY)
    return rewards


def length_reward(completions, solution, **kwargs):
    '''
    Length reward
    If the prediction is more than N times the length of the reference, -1.0 is awarded; otherwise, no points are deducted.
    1. yes-or-no question: N=1.5
    2. multiple choice question: N= 1.5
    3. open question: N=3
    '''
    rewards = []
    CONCISE_BONUS = 0.0
    VERBOSE_PENALTY = -1.0

    for completion, sol in zip(completions, solution):
        pred_ans = extract_answer_content(completion[0]["content"])
        gt_ans = extract_answer_content(sol)

        pred_len = len(pred_ans.split())
        gt_len = len(gt_ans.split())

        if is_mcq_task(gt_ans) or is_yes_no_task(gt_ans):
            max_len = max(5, gt_len * 1.5)
        else:
            max_len = max(5, gt_len * 3)

        if pred_len > max_len:
            rewards.append(VERBOSE_PENALTY)
        else:
            rewards.append(CONCISE_BONUS)
            
    return rewards


reward_funcs_registry = {
    "accuracy": accuracy_reward,
    "format": format_reward,
    "length": length_reward
}
# ---------------------------------------------------------------------- #


def main(script_args, training_args, model_args):
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]
    training_args.eval_strategy = "no"# Force no evaluation

    if not script_args.dataset_name:
        raise ValueError("No training dataset file provided. Please specify --dataset_name.")
    data_files = {script_args.dataset_train_split: script_args.dataset_name}
    dataset = load_dataset("json", data_files=data_files)
    print(f"Loaded training dataset from JSON: {data_files}")

    QUESTION_TEMPLATE_LOCAL = "{Question} Your task: 1. Think through the question step by step, enclose your reasoning process in <think>...</think> tags. 2. Provide the correct answer inside <answer>...</answer> tags. 2. No extra information or text outside of these tags."

    def make_conversation_image(example):
        """
        Create the prompt structure for the model
        """
        image_path_in_dataset = example["image"]
        full_image_path = os.path.join(script_args.image_root_dir, image_path_in_dataset)

        prompt = [
            {"role": "user", "content": [
                {"type": "image"}, 
                {"type": "text", "text": QUESTION_TEMPLATE_LOCAL.format(Question=example["problem"])}
            ]}
        ]

        return {
            "prompt": prompt,
            "image": full_image_path,
            "image_path": full_image_path,
            "solution": example["solution"],
            "problem": example["problem"],
            # Include any other fields needed by reward functions
            **{k: v for k, v in example.items() if k not in ["prompt", "image", "solution", "problem"]}
        }

    train_split_features = dataset[script_args.dataset_train_split].features
    if "image" in train_split_features:
        dataset = dataset.map(make_conversation_image)
    else:
        raise ValueError(f"Dataset must contain an 'image' column. Check your data, please!")

    trainer_cls = Qwen2VLGRPOTrainer
    print("using: ", trainer_cls)

    # KL divergence
    adaptive_kl_callback = AdaptiveKLCallback(
        initial_beta=script_args.initial_beta,
        final_beta=script_args.final_beta
    )

    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        peft_config=get_peft_config(model_args),
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
        bad_case_log_file=script_args.bad_case_log_file,
        openset_answer_pool=CATEGORIES,
        callbacks=[adaptive_kl_callback],
    )
    
    trainer.train()
    trainer.save_bad_cases()
    trainer.save_model(training_args.output_dir)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)