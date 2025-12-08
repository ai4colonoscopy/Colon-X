import os
import json
import re 
import random 
from collections import defaultdict
from typing import Any, Callable, Optional, Union, List, Dict

import torch
import torch.utils.data
import torch.distributed as dist
import transformers
from datasets import Dataset, IterableDataset
from packaging import version
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoProcessor,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    Trainer,
    TrainerCallback,
    is_wandb_available,
)
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.utils import is_peft_available

from trl.data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template
from trl.models import create_reference_model, prepare_deepspeed, unwrap_model_for_generation
from trl.trainer.grpo_config import GRPOConfig

import copy
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


if is_peft_available():
    from peft import PeftConfig, get_peft_model

if is_wandb_available():
    import wandb

RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]


class Qwen2VLGRPOTrainer(Trainer):

    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        args: GRPOConfig = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        reward_processing_classes: Optional[Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        peft_config: Optional["PeftConfig"] = None,
        max_pixels: Optional[int] = 12845056,
        min_pixels: Optional[int] = 3136,
        bad_case_log_file: Optional[str] = None,
        attn_implementation: str = "flash_attention_2",
        openset_answer_pool: Optional[List[str]] = None,
    ):
        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path
            model_name = model_name.split("/")[-1]
            args = GRPOConfig(f"{model_name}-GRPO")

        # Open question answer pool
        self.openset_answer_pool = openset_answer_pool
        if openset_answer_pool:
            self.openset_answer_pool_lower_set = {cat.lower() for cat in openset_answer_pool}
        
        model_init_kwargs = args.model_init_kwargs or {}
        model_init_kwargs["attn_implementation"] = attn_implementation
        if isinstance(model, str):
            model_id = model
            torch_dtype = model_init_kwargs.get("torch_dtype")
            if isinstance(torch_dtype, torch.dtype) or torch_dtype == "auto" or torch_dtype is None:
                pass  # torch_dtype is already a torch.dtype or "auto" or None
            elif isinstance(torch_dtype, str):  # it's a str, but not "auto"
                torch_dtype = getattr(torch, torch_dtype)
                model_init_kwargs["torch_dtype"] = torch_dtype
            else:
                raise ValueError(
                    "Invalid `torch_dtype` passed to `GRPOConfig`. Expected either 'auto' or a string representing "
                    f"a `torch.dtype` (e.g., 'float32'), but got {torch_dtype}."
                )
            # Disable caching if gradient checkpointing is enabled (not supported)
            model_init_kwargs["use_cache"] = (
                False if args.gradient_checkpointing else model_init_kwargs.get("use_cache")
            )
            if "Qwen2-VL" in model_id:
                model = Qwen2VLForConditionalGeneration.from_pretrained(model, **model_init_kwargs)
            elif any(name in model_id for name in ["ColonR1", "Qwen2.5-VL"]):
                model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model, **model_init_kwargs)
            else:
                model = AutoModelForCausalLM.from_pretrained(model, **model_init_kwargs)
        else:
            model_id = model.config._name_or_path
            if args.model_init_kwargs is not None:
                raise ValueError(
                    "You passed `model_init_kwargs` to the `GRPOConfig`, but your model is already instantiated. "
                    "This argument can only be used when the `model` argument is a string."
                )

        # LoRA 
        if peft_config is not None:
            model = get_peft_model(model, peft_config)

        # Reference model
        if is_deepspeed_zero3_enabled():
            print("DeeepSpeed is enabled.")
            if "Qwen2-VL" in model_id:
                self.ref_model = Qwen2VLForConditionalGeneration.from_pretrained(model_id, **model_init_kwargs)
            elif any(name in model_id for name in ["ColonR1", "Qwen2.5-VL"]):
                self.ref_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, **model_init_kwargs)
            else:
                self.ref_model = AutoModelForCausalLM.from_pretrained(model_id, **model_init_kwargs)
        elif peft_config is None:
            self.ref_model = create_reference_model(model)
        else:
            self.ref_model = None

        # Processing class
        if processing_class is None:
            if any(name in model_id for name in ["Qwen2-VL", "Qwen2.5-VL", "ColonR1"]):
                processing_class = AutoProcessor.from_pretrained(model_id)
                pad_token_id = processing_class.tokenizer.pad_token_id
                processing_class.pad_token_id = pad_token_id
                processing_class.eos_token_id = processing_class.tokenizer.eos_token_id
                if any(name in model_id for name in ["Qwen", "Qwen2.5-VL", "ColonR1"]):
                    processing_class.image_processor.max_pixels = max_pixels
                    processing_class.image_processor.min_pixels = min_pixels
            else:
                processing_class = AutoTokenizer.from_pretrained(model.config._name_or_path, padding_side="left")
                pad_token_id = processing_class.pad_token_id

        # Reward functions
        if not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]
        self.reward_funcs = reward_funcs

        # Reward processing class
        if reward_processing_classes is None:
            self.reward_processing_classes = [None] * len(reward_funcs)
        else:
            if len(reward_processing_classes) != len(reward_funcs):
                raise ValueError("Reward processing classes length mismatch.")
            self.reward_processing_classes = reward_processing_classes

        # Data collator
        def data_collator(features):  # No data collation is needed in GRPO
            return features

        # Training arguments
        self.max_prompt_length = args.max_prompt_length
        self.max_completion_length = args.max_completion_length
        self.num_generations = args.num_generations
        self.generation_config = GenerationConfig(
            max_new_tokens=self.max_completion_length,
            do_sample=True,
            temperature=1, # 1
            num_return_sequences=self.num_generations,
            pad_token_id=pad_token_id,
        )
        self.beta = args.beta

        self.bad_case_log_file = bad_case_log_file
        self._bad_cases_map = {}

        model.warnings_issued["estimate_tokens"] = True

        # Initialize the metrics
        self._metrics = defaultdict(list)

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
        )

        self.model_accepts_loss_kwargs = False

        if self.ref_model is not None:
            if self.is_deepspeed_enabled:
                self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                self.reward_funcs[i] = self.accelerator.prepare_model(reward_func, evaluation_mode=True)

# ------------------------- Negative sampling helper function ----------------------------#
    def extract_answer_content(self, text):
        """Extract content from the <answer> tag"""
        if not isinstance(text, str): return ""
        match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL | re.IGNORECASE)
        return match.group(1).strip() if match else text.strip()

    def get_mcq_label(self, text):
        """Extract letter form multiple choice question"""
        if not text:
            return None
        text = text.strip()
        match = re.match(r'^[<\(\[]?\s*([A-Z])\s*[>\)\]\.]?', text, re.IGNORECASE)
        if match:
            remaining_text = text[match.end():].strip()
            if not remaining_text or not remaining_text[0].isalpha():
                return match.group(1).upper()
        if len(text) == 1 and 'A' <= text.upper() <= 'Z':
            return text.upper()   
        return None
    
    def is_mcq_task(self, text):
        """Determine if it is a multiple choice question."""
        return self.get_mcq_label(text) is not None

    def is_yes_no_task(self, text):
        '''Determine if it is a yes-or-no question.'''
        return text.strip().lower() in ["yes", "no"]

    def classify_task_type(self, text):
        '''Confirm task type'''
        text_content = self.extract_answer_content(text)
        if self.is_mcq_task(text_content):
            return 'multiple_choice'
        elif self.is_yes_no_task(text_content):
            return 'yes_no'
        else:
            return 'open_set'

    def generate_replacement_answer(self, question_type, prompt_text, gt, pred):
        '''
        Generate an answer that differs from the reference to prevent the variance in the reward calculation from being 0.
        '''
        # Get the lowercase version of the reference
        gt_content_lower = self.extract_answer_content(gt).lower()

        if question_type == 'yes_no':
            '''If the model generates 'yes', it returns 'No', and vice versa.'''
            first_gen_lower = pred[0].lower() if pred else ""
            return 'No' if 'yes' in first_gen_lower else 'Yes'
        
        elif question_type == 'multiple_choice':
            all_options = re.findall(r'<([A-Z])>', prompt_text)
            if not all_options: 
                return None
            # Excluding the standard answer
            items_to_exclude = set()
            gt_label = self.get_mcq_label(self.extract_answer_content(gt))
            if gt_label:
                items_to_exclude.add(gt_label)
            # Exclude options that have already been generated
            for content in pred:
                gen_label = self.get_mcq_label(self.extract_answer_content(content))
                if gen_label:
                    items_to_exclude.add(gen_label)
            # Choose one from the remaining options
            candidate_pool = [opt for opt in all_options if opt not in items_to_exclude]
            return random.choice(candidate_pool) if candidate_pool else None

        elif question_type == 'open_set':
            if not self.openset_answer_pool: 
                return None
            # Excluding the standard answer
            items_to_exclude = {gt_content_lower}
            # Exclude content that have already been generated
            for content in pred:
                extracted_content_lower = self.extract_answer_content(content).lower()
                items_to_exclude.add(extracted_content_lower)
            # Choose one from the remaining items in the pool
            candidate_pool = [ans for ans in self.openset_answer_pool if ans.lower() not in items_to_exclude]
            return random.choice(candidate_pool) if candidate_pool else None
            
        return None
# -----------------------------------------------------------------------------------------#


    def _set_signature_columns_if_needed(self):
        if self._signature_columns is None:
            self._signature_columns = ["prompt"]


    def _get_per_token_logps(self, model, input_ids, attention_mask, pixel_values, image_grid_thw):
        logits = model(input_ids, attention_mask=attention_mask, pixel_values=pixel_values, image_grid_thw=image_grid_thw).logits
        logits = logits[:, :-1, :]
        input_ids = input_ids[:, 1:]
        per_token_logps = []
        for logits_row, input_ids_row in zip(logits, input_ids):
            log_probs = logits_row.log_softmax(dim=-1)
            token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)
            per_token_logps.append(token_log_prob)
        return torch.stack(per_token_logps)


    def _prepare_inputs(self, inputs):
        '''This prevented the Trainer from automatically moving data to the GPU'''
        return inputs


    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")
        
        # Input preprocessing and image loading
        # Extract the question and format it using a chat template.
        prompts = [x["prompt"] for x in inputs]
        prompts_text = [maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs]
        
        # Process images
        image_paths = [x["image"] for x in inputs]
        
        # resize image size to 384, reference: https://github.com/Yuxiang-Lai117/Med-R1/issues/6
        target_size = (384, 384)
        images = [Image.open(path).convert("RGB").resize(target_size) for path in image_paths]
        
        # Use the Qwen2-VL processor to process text and images.
        prompt_inputs = self.processing_class(
            text=prompts_text,
            images=images,
            return_tensors="pt",
            padding=True,
            padding_side="left",
        )
        prompt_inputs = super()._prepare_inputs(prompt_inputs)
        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]
        pixel_values = prompt_inputs["pixel_values"]
        image_grid_thw = prompt_inputs["image_grid_thw"]
        
        # If the prompt is too long, it will be truncated.
        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length :]
            prompt_mask = prompt_mask[:, -self.max_prompt_length :]
        
        # Generate completions
        with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
            prompt_completion_ids = unwrapped_model.generate(**prompt_inputs, generation_config=self.generation_config, use_model_defaults=False)
        
        prompt_length = prompt_ids.size(1)
        completion_ids = prompt_completion_ids[:, prompt_length:]
        prompt_mask = prompt_mask.repeat_interleave(self.num_generations, dim=0)

        # Mask everything after the first EOS token
        is_eos = completion_ids == self.processing_class.eos_token_id
        device = self.accelerator.device
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        # Concatenate prompt_mask with completion_mask for logit computation
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        pixel_values = prompt_inputs["pixel_values"].repeat(self.num_generations, 1, 1, 1)
        image_grid_thw = prompt_inputs["image_grid_thw"].repeat_interleave(self.num_generations, dim=0)

        per_token_logps = self._get_per_token_logps(model, prompt_completion_ids, attention_mask, pixel_values, image_grid_thw)
        # Get rid of the prompt (-1 because of the shift done in get_per_token_logps)
        per_token_logps = per_token_logps[:, prompt_length - 1 :]

        with torch.inference_mode():
            if self.ref_model is not None:
                ref_per_token_logps = self._get_per_token_logps(self.ref_model, prompt_completion_ids, attention_mask, pixel_values, image_grid_thw)
            else:
                with self.accelerator.unwrap_model(model).disable_adapter():
                    ref_per_token_logps = self._get_per_token_logps(model, prompt_completion_ids, attention_mask, pixel_values, image_grid_thw)
        ref_per_token_logps = ref_per_token_logps[:, prompt_length - 1 :]

        # Compute the KL divergence between the model and the reference model
        per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1

        # Decode the generated completions
        completions = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        if is_conversational(inputs[0]):
            completions = [[{"role": "assistant", "content": completion}] for completion in completions]

        # Compute the rewards
        prompts = [prompt for prompt in prompts for _ in range(self.num_generations)]

        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)
        for i, (reward_func, reward_processing_class) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes)
        ):
            if isinstance(reward_func, PreTrainedModel):
                if is_conversational(inputs[0]):
                    messages = [{"messages": p + c} for p, c in zip(prompts, completions)]
                    texts = [apply_chat_template(x, reward_processing_class)["text"] for x in messages]
                else:
                    texts = [p + c for p, c in zip(prompts, completions)]
                reward_inputs = reward_processing_class(
                    texts, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False
                )
                reward_inputs = super()._prepare_inputs(reward_inputs)
                with torch.inference_mode():
                    rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]
            else:
                reward_kwargs = {key: [] for key in inputs[0].keys() if key not in ["prompt", "completion"]}
                for key in reward_kwargs:
                    for example in inputs:
                        reward_kwargs[key].extend([example[key]] * self.num_generations)
                
                # Pass num_generations_per_prompt for debug output in the training code.
                reward_kwargs["num_generations_per_prompt"] = self.num_generations
                output_reward_func = reward_func(prompts=prompts, completions=completions, **reward_kwargs)
                rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

        # Sum the rewards from all reward functions
        rewards = rewards_per_func.sum(dim=1)

        # --------------------------- Negative sampling ---------------------------------------
        num_prompts_in_batch = len(inputs)
        for i in range(num_prompts_in_batch):
            start_idx = i * self.num_generations
            end_idx = start_idx + self.num_generations
            item_rewards = rewards[start_idx:end_idx]

            # Rewards same within a group, randomly replace one.
            if self.num_generations > 1 and len(set(item_rewards.tolist())) == 1:
                local_idx_to_replace = random.randrange(self.num_generations)
                absolute_idx_to_replace = start_idx + local_idx_to_replace

                # Extract prompt text
                prompt_info = prompts[start_idx]
                prompt_text = ""
                user_prompt = next((p for p in prompt_info if p['role'] == 'user'), None)
                if user_prompt and isinstance(user_prompt['content'], list):
                    text_content = next((c['text'] for c in user_prompt['content'] if c['type'] == 'text'), "")
                    prompt_text = text_content
                
                # Get the standard answer
                gt_answer = inputs[i]["solution"]

                # Get all generated content in the current group
                item_completions = completions[start_idx:end_idx]
                all_generated_contents = [comp[0]["content"] for comp in item_completions]
                
                # get task type and a negative sample
                task_type = self.classify_task_type(gt_answer)
                replacement_answer = self.generate_replacement_answer(question_type=task_type,
                                                                      prompt_text=prompt_text,
                                                                      gt=gt_answer,
                                                                      pred=all_generated_contents)
                # Replace original response
                if replacement_answer:
                    # Preserve the original content and rewards. (for debug only)
                    original_content = completions[absolute_idx_to_replace][0]["content"]
                    original_reward = rewards[absolute_idx_to_replace].item()

                    # Get the original, complete content and replace only the content within the <answer> tag.
                    original_full_content = completions[absolute_idx_to_replace][0]["content"]
                    think_match = re.search(r'(<think>.*?</think>)', original_full_content, re.DOTALL | re.IGNORECASE)
                    answer_match = re.search(r'<answer>.*?</answer>', original_full_content, re.DOTALL | re.IGNORECASE)
                    new_answer_tag = f"<answer>{replacement_answer}</answer>"
                    if think_match and answer_match: # answer & reasoning
                        new_full_content = think_match.group(1) + new_answer_tag
                        print(f"[Debug Replacement Rank {self.accelerator.process_index}] Preserved <think>, replaced <answer>.")
                    elif answer_match: # Only answers, no reasoning.
                        new_full_content = new_answer_tag
                        print(f"[Debug Replacement Rank {self.accelerator.process_index}] Only <answer> found, replaced.")
                    else:
                        new_full_content = new_answer_tag
                        print(f"[Debug Replacement Rank {self.accelerator.process_index}] No <answer> found in original, using new one.")
                    # Write back to the Completions list
                    completions[absolute_idx_to_replace][0]["content"] = new_full_content

                    # Recalculate reward
                    with torch.inference_mode():
                        for rf_idx, (reward_func, reward_processing_class) in enumerate(
                            zip(self.reward_funcs, self.reward_processing_classes)
                        ):
                            single_reward_kwargs = {}
                            for key, val in inputs[i].items():
                                if key in ["prompt", "completion"]:
                                    continue
                                single_reward_kwargs[key] = [val]
                                single_reward_kwargs["num_generations_per_prompt"] = 1
                            # Extract the modified single data.
                            single_prompt_obj = prompts[absolute_idx_to_replace]
                            single_completion_obj = completions[absolute_idx_to_replace]
                            
                            # Call  reward function to recalculate.
                            output = reward_func(
                                prompts=[single_prompt_obj], 
                                completions=[single_completion_obj],
                                **single_reward_kwargs
                            )
                            if isinstance(output, (list, tuple)):
                                val = float(output[0])
                            else:
                                val = float(output)
                            rewards_per_func[absolute_idx_to_replace, rf_idx] = torch.tensor(val, dtype=torch.float32, device=device)
                    
                    # Update total score
                    rewards[absolute_idx_to_replace] = rewards_per_func[absolute_idx_to_replace].sum()

                    # # [Debug] To view detailed information about the training process, you can uncomment the following code. 
                    # print("\n" + "="*30 + f" ADAPTIVE REPLACEMENT & RE-SCORE ON RANK {self.accelerator.process_index} " + "="*30)
                    # print(f"[Debug] prompt: {prompt_text}")
                    # print(f"-> GT: {gt_answer.strip()}")
                    # print(f"-> All rewards were identical: {item_rewards[0].item()}")
                    # print("-" * 35)
                    # print("Original Generations & Rewards (before replacement):")
                    # # Loop to print all original generations
                    # for j in range(self.num_generations):
                    #     original_idx = start_idx + j
                    #     original_generation_reward = item_rewards[j].item()
                    #     if original_idx == absolute_idx_to_replace:
                    #         original_generation_content = original_content
                    #     else:
                    #         original_generation_content = completions[original_idx][0]["content"]
                    #     print(f"  -> generation_{j+1}: {original_generation_content.strip()}")
                    #     print(f"  -> reward_{j+1}    : {original_generation_reward}")
                    # print("-" * 35)
                    # print("Replacement & Re-score Details:")
                    # print(f"-> Replacing generation at local index: {local_idx_to_replace + 1}")
                    # print(f"-> Original content for replacement: {original_content.strip()}")
                    # print(f"-> New replaced content: {new_full_content}")
                    # print(f"-> New replaced reward (sum over funcs): {rewards[absolute_idx_to_replace].item()}")
                    # per_func_vals = rewards_per_func[absolute_idx_to_replace].tolist()
                    # print(f"-> Per-reward-func scores: {per_func_vals}")
                    # print(f"-> Task Type: {task_type}")
                    # print("="*88 + "\n")
                    
        # ---------------------------------------------------------------------------------------------- #

        # ---------------------------------------- Self-evolving --------------------------------------- #
        if self.bad_case_log_file is not None:

            # Extract the score for each group of answers.
            for i in range(num_prompts_in_batch):
                final_item_rewards = [rewards[i * self.num_generations + j].item() for j in range(self.num_generations)]
                
                # Determine if it is a bad case, i.e., whether all reward < 0.8.
                if self.num_generations > 0 and all(r < 0.8 for r in final_item_rewards):
                    # Get data (Q/A/Img) from the original inputs.
                    problem_text = inputs[i].get("problem", "N/A")
                    solution_text = inputs[i].get("solution", "N/A")
                    image_path = inputs[i].get("image", "image_path_not_found_in_data")
                    dict_key = (image_path, problem_text)

                    # If it is not recorded, record it
                    if dict_key not in self._bad_cases_map:
                        final_completions_for_item = [
                            completions[i * self.num_generations + j][0]["content"]
                            for j in range(self.num_generations)
                        ]
                        self._bad_cases_map[dict_key] = {
                            "image": image_path,
                            "problem": problem_text,
                            "solution": solution_text,
                            "wrong_pred": final_completions_for_item
                        }
        # ---------------------------------------------------------------------------------------------- #

        # Calculate the mean & standard deviation within the group
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)
        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        # Calculate advantage: advantage = (score - mean) / (standard deviation + epsilon)
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)

        # x - x.detach() allows for preserving gradients from x
        per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1)
        per_token_loss = -(per_token_loss - self.beta * per_token_kl)
        loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()

        # Log the metrics
        completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
        self._metrics["completion_length"].append(completion_length)

        reward_per_func = self.accelerator.gather_for_metrics(rewards_per_func).mean(0)
        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                reward_func_name = reward_func.config._name_or_path.split("/")[-1]
            else:
                reward_func_name = reward_func.__name__
            self._metrics[f"rewards/{reward_func_name}"].append(reward_per_func[i].item())

        self._metrics["reward"].append(self.accelerator.gather_for_metrics(rewards).mean().item())
        self._metrics["reward_std"].append(self.accelerator.gather_for_metrics(std_grouped_rewards).mean().item())

        mean_kl = ((per_token_kl * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
        self._metrics["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())

        return loss


    def save_bad_cases(self):
        '''
        Save bad case in one file
        '''
        # If the bad_case path is not specified, no recording is required by default, and the program will exit directly.
        if not self.bad_case_log_file:
            return
        
        # Collect data from each GPU
        local_bad_cases = list(self._bad_cases_map.values())
        if not dist.is_initialized():
            gathered_cases_lists = [local_bad_cases]
        else:
            gathered_cases_lists = [None] * self.accelerator.num_processes
            dist.all_gather_object(gathered_cases_lists, local_bad_cases)
        # Deduplication & save
        if self.is_world_process_zero():
            print(f"Aggregating and saving bad cases from all processes to {self.bad_case_log_file}...")
            final_merged_map = {}
            for process_cases in gathered_cases_lists:
                for case in process_cases:
                    key = (case["image"], case["problem"])
                    if key in final_merged_map:
                        final_merged_map[key]["wrong_pred"].extend(case["wrong_pred"])
                    else:
                        final_merged_map[key] = case
            final_bad_cases = list(final_merged_map.values())
            output_dir = os.path.dirname(self.bad_case_log_file)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            with open(self.bad_case_log_file, "w", encoding="utf-8") as f:
                json.dump(final_bad_cases, f, indent=4, ensure_ascii=False)
            print(f"Successfully saved {len(final_bad_cases)} unique bad case items.")
            self._bad_cases_map.clear()
            print("Bad cases map has been cleared for the next logging interval.")



    def log(self, logs: dict[str, float], start_time: Optional[float] = None):
        metrics = {key: sum(val) / len(val) for key, val in self._metrics.items()}
        logs = {**logs, **metrics}
        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super().log(logs, start_time)
        else:
            super().log(logs)
        self._metrics.clear()

