import json
import re
import argparse
from typing import List, Tuple, Optional, Dict, Any, Union
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Define the number of each task (all)
EXPECTED_COUNTS = {
    1: 2196,
    2: 3308,
    3: 416,
    4: 991,
    5: 1140,
    6: 5110,
    7: 1006,
    8: 1224,
    9: 90492,
    10: 45246,
    11: 45246,
    12: 50865, 
    13: 50865,
    14: 310,
    15: 29592,
    16: 29775,
    17: 2015
}


# Define the number of each task (pilot)
EXPECTED_COUNTS_PILOT = {
    1: 50,
    2: 50,
    3: 50,
    4: 50,
    5: 50,
    6: 71,
    7: 50,
    8: 50,
    9: 1258,
    10: 629,
    11: 629,
    12: 707, 
    13: 707,
    14: 50,
    15: 410,
    16: 414,
    17: 50
}

# Define the number of each task (pert)
EXPECTED_COUNTS_PERT = {
    "A": 20,
    "B": 100,
    "C": 57,
    "D": 80
}


# Prompt used when using LLM as a judge in a yes-or-no task
YES_NO_ARBITRATION_PROMPT = """
You are an impartial judge. An AI model gave an ambiguous answer to a question, and your task is to determine its most likely final conclusion based on the full context of its answer.
The original question asked for a "yes" or "no" answer. Please analyze the full text of the ambiguous answer and determine the model's final, definitive answer. You must choose one and only one of the following options: "yes" or "no". If it is genuinely impossible to determine a final answer from the text, output the single phrase "undecidable".
Do not provide any explanation, reasoning, or additional text. Your output must be a single word "yes", "no" or "undecidable".

Ambiguous answer: "{ambiguous_text}" 
"""


# Prompt used when using LLM as a judge in a single choice task
SCQ_ARBITRATION_PROMPT = """
You are an impartial judge. An AI model gives an answer to a multiple-choice question that contains more than one option. Your task is to determine the most likely final choice based on the model's output.
Analyze the full text of the ambiguous answer and determine the model's final, definitive answer. You must choose one and only one of the options provided. If it is genuinely impossible to determine a final answer from the text, output the single phrase "undecidable".
Do not provide any explanation or extra text. Your output must be either one of the options or the phrase "undecidable".

Original question:
"{question_text}"
Ambiguous answer:
"{ambiguous_text}"
"""


# Call the LLM as the judge to get the exact answer
def call_llm_as_judge(prompt, gpt_oss_resources):

    if gpt_oss_resources is None:
        print("[Warning] gpt-oss resources are not loaded.")
        return None
    try:
        model = gpt_oss_resources["model"]
        tokenizer = gpt_oss_resources["tokenizer"]
        
        messages = [{"role": "user", "content": prompt}]
        
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)

        outputs = model.generate(**inputs, max_new_tokens=30)
        clarified_answer = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True).strip()
        return clarified_answer, None
    except Exception as e:
        error_message = str(e)
        print(f"[Warning] LLM call error (gpt-oss): {error_message}")
        return None, error_message


def normalize_text(text):
    """
    Normalize the text by removing special characters and extra spaces. (used for open-ended VQA tasks)
    Replace '-', '/' and ' ' with spaces and remove extra punctuation
    """
    if not text:
        return ""
    text = re.sub(r'[-/]', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()


# IoU calculation
def compute_iou(bbox1, bbox2):
    '''reference 
    https://github.com/ai4colonoscopy/IntelliScope/blob/main/script/multimodal_benchmark/multimodal_evaluator.py#L46
    '''
    if len(bbox1) != 4 or len(bbox2) != 4:
        return 0.0

    xmin1, ymin1, xmax1, ymax1 = bbox1
    xmin2, ymin2, xmax2, ymax2 = bbox2

    inter_xmin = max(xmin1, xmin2)
    inter_ymin = max(ymin1, ymin2)
    inter_xmax = min(xmax1, xmax2)
    inter_ymax = min(ymax1, ymax2)

    inter_width = max(0, inter_xmax - inter_xmin + 1)
    inter_height = max(0, inter_ymax - inter_ymin + 1)

    inter_area = inter_width * inter_height
    if inter_area == 0:
        return 0.0

    bbox1_area = (xmax1 - xmin1 + 1) * (ymax1 - ymin1 + 1)
    bbox2_area = (xmax2 - xmin2 + 1) * (ymax2 - ymin2 + 1)
    
    union_area = float(bbox1_area + bbox2_area - inter_area)
    if union_area == 0:
        return 0.0

    iou = inter_area / union_area
    return iou


def parse_range_conditions(text):
    """Parses range conditions from a text string. 
    This function can handle various formats, including:
    * '0mm < polyp < 6mm'
    * '6mm ≤ polyp < 20mm'
    * 'polyp ≥ 30mm'

    The function standardizes the conditions and returns a tuple with two elements:
    - The word extracted from the text (e.g., 'polyp').
    - A list of conditions, formatted as [(operator, value), ...].

    Sample, '6mm ≤ polyp < 20mm' return 'polyp', [('>=', '6'), ('<', '20')]
    """
    text = text.replace('≥', '>=').replace('≤', '<=').replace('mm', '')
    word = re.search(r'\b([a-zA-Z]+)\b', text)
    word = word.group(1)

    pattern = r'([><]=?)\s*(\d+)|(\d+)\s*([><]=?)'
    matches = re.findall(pattern, text)
    conditions = []

    for op1, num1, num2, op2 in matches:
        if op1 and num1:
            conditions.append((op1, num1))
        elif op2 and num2:
            # let 'polyp' be the subject
            op_map = {'>': '<', '<': '>', '>=': '<=', '<=': '>='}
            conditions.append((op_map[op2], num2))
            
    return word, conditions


def parse_answer(text):
    """Parse various formats for single-choice VQA tasks, including:
    * <A> Answer Text
    * <A> Answer Text (Abbreviation)
    * <A> Answer Text (characteristic for xxxxx)

    Returns a tuple of three elements:
    - The option letter (e.g., 'A')
    - The main answer text
    - The alias or additional descriptor (if any), which is either the content in parentheses
      or the text following 'characteristic for'.
    """

    # match = re.match(r"<([A-Z])>\s*(.*?)(?:\s*\((.*)\))?\s*$", text, re.DOTALL)
    match = re.match(r"<([A-Z])>(.*?)(?:\s*\((.*)\))?\s*$", text, re.DOTALL)

    if not match:
        return None

    letter, content, extra_text = match.groups()

    letter = letter.strip()
    content = content.strip()
    abbr = None

    if extra_text:
        prefix = 'characteristic for '
        abbr_stripped = extra_text.strip()
        if abbr_stripped.startswith(prefix):
            abbr = abbr_stripped[len(prefix):].strip()
        else:
            abbr = abbr_stripped
            
    return letter, content, abbr


def parse_option(text):
    text = text.strip()
    abbr_match = re.search(r'\((.*?)\)', text)
    
    if abbr_match:
        abbr = abbr_match.group(1).strip().lower()
        main_text = text[:abbr_match.start()].strip().lower()
        return main_text, abbr
    else:
        return text.lower(), None


def is_invalid(prediction, available_options):
    """
    Checks whether the predictions are from outside the question options.
    Returns True if so, otherwise returns False.
    """
    # If the format is incorrect, skip.
    parsed_pred = parse_answer(prediction)
    if not parsed_pred:
        return False

    pred_letter, pred_content, pred_abbr = parsed_pred

    if not pred_content and not pred_abbr:
        return False

    # Create a list of options
    option_letters = [opt[0] for opt in available_options]
    parsed_options = [parse_option(opt[1]) for opt in available_options]
    option_content = [opt[0] for opt in parsed_options]
    option_abbrs = [opt[1] for opt in parsed_options if opt[1] is not None]

    if pred_letter not in option_letters:
        return True

    # Check if content is valid
    if pred_content:
        content_is_valid = False
        pred_content_lower = pred_content.lower()
        for opt_cont in option_content:
            if pred_content_lower in opt_cont or opt_cont in pred_content_lower:
                content_is_valid = True
                break
        if not content_is_valid:
            return True

    # Check if abbr is valid
    if pred_abbr:
        abbr_is_valid = False
        pred_abbr_lower = pred_abbr.lower()
        for opt_abbr in option_abbrs:
            if pred_abbr_lower in opt_abbr or opt_abbr in pred_abbr_lower:
                abbr_is_valid = True
                break
        if not abbr_is_valid:
            return True

    return False


def is_mixed_mismatch(prediction, available_options):
    """
    Checks if there is a mismatch between the prediction option letters and the content. 
    Returns True if so, otherwise returns False.
    """
    # If the format is incorrect, skip.
    parsed_pred = parse_answer(prediction)
    if not parsed_pred:
        return False

    # Create a list of options
    pred_letter, pred_content, pred_abbr = parsed_pred
    option_letters = [opt[0] for opt in available_options]
    parsed_options = [parse_option(opt[1]) for opt in available_options]
    option_content = [opt[0] for opt in parsed_options]
    option_abbrs = [opt[1] for opt in parsed_options]

    # Get the index of each part
    def find_index(item, lst):
        try:
            return lst.index(item) if item else -1
        except ValueError:
            return -1
    
    letter_idx = find_index(pred_letter, option_letters)
    text_idx = find_index(pred_content.lower(), option_content)
    extra_idx = find_index(pred_abbr.lower() if pred_abbr else None, option_abbrs)

    # Check if there is a mixmatch
    found_indices = {idx for idx in [letter_idx, text_idx, extra_idx] if idx != -1}
    if len(found_indices) > 1:
        return True

    return False


def is_multiple_options(prediction, available_options):
    """
    Check how many different options were mentioned in the prediction.
    """

    # Create a list of options
    option_letters = [opt[0].lower() for opt in available_options]
    parsed_options = [parse_option(opt[1]) for opt in available_options]
    option_content = [opt[0] for opt in parsed_options]
    option_abbrs = [opt[1] for opt in parsed_options]
    
    # Create a collection to store the indexes of found options
    found_matches = []

    # Find all possible content matches
    for i, text in enumerate(option_content):
        if text and re.search(rf"\b{re.escape(text)}\b", prediction, re.IGNORECASE):
            found_matches.append({'index': i, 'text': text, 'type': 'content'})
    
    # Find all possible letter matches
    for i, letter in enumerate(option_letters):
        pattern = rf"(?<![a-zA-Z]')(?<![a-zA-Z0-9]){re.escape(letter)}(?![a-zA-Z0-9])"
        if re.search(pattern, prediction, re.IGNORECASE):
            found_matches.append({'index': i, 'text': letter, 'type': 'letter'})

    # Find all possible abbreviation matches
    for i, abbr in enumerate(option_abbrs):
        if abbr and re.search(rf"\b{re.escape(abbr)}\b", prediction, re.IGNORECASE):
            found_matches.append({'index': i, 'text': abbr, 'type': 'abbr'})

    # Deduplicate and filter out substrings
    if not found_matches:
        return 0

    # Sort by text length descending, so we process longer matches first
    sorted_matches = sorted(found_matches, key=lambda x: len(x['text']), reverse=True)
    
    final_indices = set()
    
    for match in sorted_matches:
        is_substring_of_existing = False
        # Check if this match is a substring of an already-added longer match
        for existing_index in final_indices:
            # We only care about content-in-content substrings
            existing_text = option_content[existing_index]
            if match['type'] == 'content' and match['text'] in existing_text and match['text'] != existing_text:
                is_substring_of_existing = True
                break
        
        if not is_substring_of_existing:
            final_indices.add(match['index'])

    return len(final_indices)


def eval_single_choice_vqa(item, gpt_oss_resources, cleaned_pred=None):
    ground_truth = None
    prediction = None
    prediction_index = -1
    human_question = ""

    # parse conversations content
    for i, conv in enumerate(item.get("conversations", [])):
        if conv.get("from") == "gpt":
            ground_truth = conv.get("value")
        elif conv.get("from") == "prediction":
            prediction = conv.get("value")
            prediction_index = i
        elif conv.get("from") == "human":
            human_question = conv.get("value")

    if cleaned_pred is not None:
        prediction = cleaned_pred.strip()
    elif prediction is not None:
        prediction = prediction.strip()
    else:
        print('[Warning] LLM prediction is empty, skipping evaluation for this item.')
        return 0.0

    available_options = re.findall(r"<([A-Z])>\s*(.*?)(?=\s*<[A-Z]>|$)", human_question)
    
    pred_lower = prediction.strip().lower()
    if is_invalid(prediction, available_options):
        return 0.0
    if is_mixed_mismatch(prediction, available_options):
        return 0.0
    options_found_count = is_multiple_options(pred_lower, available_options)

    if options_found_count > 1:
        prompt = SCQ_ARBITRATION_PROMPT.format(
            question_text=human_question.replace("<image>\n", ""),
            ambiguous_text=prediction
        )
        clarified_pred, error_msg = call_llm_as_judge(prompt, gpt_oss_resources)
        if clarified_pred:
            item["conversations"][prediction_index]["from"] = "prediction_ori"
            item["conversations"].append({
                "from": "prediction",
                "value": clarified_pred
            })
            prediction = clarified_pred
        else:
            print(f"error: {error_msg}")
            print(f'We encountered an ambiguous answer for current task, but LLM initialization/arbitration failed. We are skipping this item, while we really suggest you enable LLM-as-a-judge for reasonable evaluation results.')
            return 0.0
            
    final_pred_lower = prediction.strip().lower()
    # award 0.0 for "undecidable" predictions
    if "undecidable" in final_pred_lower:
        return 0.0

    parsed_gt = parse_answer(ground_truth)

    gt_letter_orig, gt_content, gt_abbr = parsed_gt
    gt_letter = gt_letter_orig.lower() 

    # We consider four types of option matching. Thus, the following types will be awarded as 1.0: <A> (A) A A. A> A) A] <A (A {A
    option_types = (
        rf"<{gt_letter}>"  # <A>
        rf"|(\({gt_letter}\))"  # (A)
        rf"|^{gt_letter}$"  # A
        rf"|\b{re.escape(gt_letter)}[\s.)>\]]"  # A. A) A> A]
        rf"|[\s<([]({re.escape(gt_letter)})\b"  # <A (A <A [A
    )

    # score 1.0 if the option letter matches
    if re.search(option_types, final_pred_lower, re.IGNORECASE):
        return 1.0
    # score 1.0 if the option content matches
    if gt_content and gt_content.lower() in final_pred_lower:
        return 1.0
    if gt_abbr and gt_abbr.lower() in final_pred_lower:
        return 1.0
    return 0.0


def eval_numerical_range(item, gpt_oss_resources, cleaned_pred=None):
    ground_truth = None
    prediction = None
    prediction_index = -1
    human_question = ""

    # parse conversations content
    for i, conv in enumerate(item.get("conversations", [])):
        if conv.get("from") == "gpt":
            ground_truth = conv.get("value")
        elif conv.get("from") == "prediction":
            prediction = conv.get("value")
            prediction_index = i
        elif conv.get("from") == "human":
            human_question = conv.get("value")

    # if pred is empty, the score is 0
    if cleaned_pred is not None:
        prediction = cleaned_pred.strip()
    elif prediction is not None:
        prediction = prediction.strip()
    else:
        print('[Warning] LLM prediction is empty, skipping evaluation for this item.')
        return 0.0

    # get all options from the question
    available_options = re.findall(r"<([A-Z])>\s*(.*?)(?=\s*<[A-Z]>|$)", human_question)

    pred_lower = prediction.strip().lower()

    # to check if the prediction is invalid
    if is_invalid(prediction, available_options):
        return 0.0

    # check if mismatch option label and content
    if is_mixed_mismatch(prediction, available_options):
        return 0.0

    # to check if the prediction contains multiple options
    options_found_count = is_multiple_options(pred_lower, available_options)

    # if so, call llm as judge
    if options_found_count > 1:
        prompt = SCQ_ARBITRATION_PROMPT.format(
            question_text=human_question.replace("<image>\n", ""),
            ambiguous_text=prediction
        )
        clarified_pred, error_msg = call_llm_as_judge(prompt, gpt_oss_resources)

        if clarified_pred:
            item["conversations"][prediction_index]["from"] = "prediction_ori"
            item["conversations"].append({
                "from": "prediction",
                "value": clarified_pred
            })
            prediction = clarified_pred
        else:
            print(f"error: {error_msg}")
            print(f'We encountered an ambiguous answer for current task, but LLM initialization/arbitration failed. We are skipping this item, while we really suggest you enable LLM-as-a-judge for reasonable evaluation results.')

    final_pred_lower = prediction.strip().lower()
    # award 0.0 for "undecidable" predictions
    if "undecidable" in final_pred_lower:
        return 0.0
        
    gt_letter, gt_content, gt_abbr = parse_answer(ground_truth)

    # We consider four types of option matching. Thus, the following types will be awarded as 1.0: <A> (A) A A. A> A) A] <A (A {A
    option_types = (
        rf"<{gt_letter}>"  # <A>
        rf"|(\({gt_letter}\))"  # (A)
        rf"|^{gt_letter}$"  # A
        rf"|\b{re.escape(gt_letter)}[\s.)>\]]"  # A. A) A> A]
        rf"|[\s<([]({re.escape(gt_letter)})\b"  # <A (A <A [A
    )
    # score 1.0 if the option letter matches
    if re.search(option_types.lower(), final_pred_lower):
        return 1.0
    # score 1.0 if the option content matches
    if gt_content.lower() in final_pred_lower:
        return 1.0

    # score 1.0 if semantic range matching
    word, conditions = parse_range_conditions(gt_content)

    all_conditions_met = True
    for op, value in conditions:
        op_patterns = {
            '>=': r'(?:at least|or more|or greater|greater than or equal to|≥)',
            '>': r'(?:larger than|greater than|more than|above|>)',
            '<=': r'(?:at most|or less|or smaller|less than or equal to|≤)',
            '<': r'(?:less than|smaller than|below|<)'
        }
        semantic_pattern = rf'\b({word}|it|is)\b.*?(?:{op_patterns[op]})\s*{value}\b'
        
        if not re.search(semantic_pattern, final_pred_lower, re.IGNORECASE):
            all_conditions_met = False
            break
    
    if all_conditions_met:
        return 1.0

    return 0.0


def eval_yes_or_no(item, gpt_oss_resources, cleaned_pred=None):
    ground_truth = None
    prediction = None
    prediction_index = -1

    # parse conversations content
    for i, conv in enumerate(item.get("conversations", [])):
        if conv.get("from") == "gpt":
            ground_truth = conv.get("value")
        elif conv.get("from") == "prediction":
            prediction = conv.get("value")
            prediction_index = i

    # if pred is empty, the score is 0
    if cleaned_pred is not None:
        prediction = cleaned_pred.strip()
    elif prediction is not None:
        prediction = prediction.strip()
    else:
        print('[Warning] LLM prediction is empty, skipping evaluation for this item.')
        return 0.0

    pred_lower = prediction.lower()
    gt_lower = ground_truth.strip().lower()

    # check if the prediction contains both "yes" and "no" words, we will call llm-as-a-judge to revise the prediction to a determined answer
    found_yes = re.search(r'\byes\b', pred_lower)
    found_no = re.search(r'\bno\b', pred_lower)

    if found_yes and found_no:
        prompt = YES_NO_ARBITRATION_PROMPT.format(ambiguous_text=prediction)
        clarified_pred, error_msg = call_llm_as_judge(prompt, gpt_oss_resources)
        if clarified_pred:
            item["conversations"][prediction_index]["from"] = "prediction_ori"
            item["conversations"].append({
                "from": "prediction",
                "value": clarified_pred
            })
            prediction = clarified_pred
        else:
            print(f"error: {error_msg}")
            print(f'We encountered an ambiguous answer for current task, but LLM initialization/arbitration failed. We are skipping this item, while we really suggest you enable LLM-as-a-judge for reasonable evaluation results.')

    final_pred_lower = prediction.strip().lower()

    # award 0.0 for "undecidable" predictions
    if "undecidable" in final_pred_lower:
        return 0.0
    # do exact match
    if re.search(r'\b' + re.escape(gt_lower) + r'\b', final_pred_lower):
        return 1.0
    
    return 0.0


def eval_anatomical_landmark(item, cleaned_pred=None):
    ground_truth = None
    prediction = None

    # parse conversations content
    for i, conv in enumerate(item.get("conversations", [])):
        if conv.get("from") == "gpt":
            ground_truth = conv.get("value")
        elif conv.get("from") == "prediction":
            prediction = conv.get("value")

    # if pred is empty, the score is 0
    if cleaned_pred is not None:
        prediction = cleaned_pred.strip()
    elif prediction is not None:
        prediction = prediction.strip()
    else:
        print('[Warning] LLM prediction is empty, skipping evaluation for this item.')
        return 0.0

    # lowercase all words for case-insensitive comparison 
    gt_lower = ground_truth.lower().strip()
    pred_lower = prediction.lower().strip()

    # if nothing is predicted, return zero
    if not pred_lower:
        return 0.0

    # do exact match
    if gt_lower == pred_lower:
        return 1.0

    decision_correct = False
    # get GT's decision
    gt_text = gt_lower.split()
    gt_decision = gt_text[0].strip(".,?!")

    # definition agreement and disagreement keyword list
    agreement_keywords = [
        'yes', 'shows', 'displays', 'confirms', 'is present', 'can be seen',
        'visible', 'identified', 'observed', 'detected', 'reveals', 'presents', 'indicates'
    ]
    disagreement_keywords = [
        'no', 'not', 'fails to show', 'does not show', 'cannot be seen', 'is absent',
        'is not present', 'not visible', 'no sign of', 'unremarkable', 'clear of', 'devoid of'
    ]
    pred_show_agreement = any(re.search(r'\b' + re.escape(key) + r'\b', pred_lower) for key in agreement_keywords)
    pred_show_disagreement = any(re.search(r'\b' + re.escape(key) + r'\b', pred_lower) for key in disagreement_keywords)

    pred_decision = "unknown"
    # model leans towards agreement
    if pred_show_agreement and not pred_show_disagreement:
        pred_decision = "yes"
    # model leans towards disagreement
    elif pred_show_disagreement and not pred_show_agreement:
        pred_decision = "no"

    if gt_decision == pred_decision:
        decision_correct = True

    # Start scoring with the correct stance.
    score = 0.0
    # If pred's decision is consistent with GT's decision, score is 0.5
    if decision_correct:
        score = 0.5

        finding_correct = False
        if "not show anatomical landmark" in gt_lower and ("not show" in pred_lower or "unclear" in pred_lower):
            finding_correct = True
        elif "shows ileocecal valve" in gt_lower and "ileocecal valve" in pred_lower:
            finding_correct = True
        elif "shows cecum" in gt_lower and "cecum" in pred_lower:
            finding_correct = True
        # If the finding also matches, the score is 1
        if finding_correct:
            score = 1.0

    return score


def eval_open_vqa(item, cleaned_pred=None):
    ground_truth = None
    prediction = None

    # parse conversations content
    for i, conv in enumerate(item.get("conversations", [])):
        if conv.get("from") == "gpt":
            ground_truth = conv.get("value")
        elif conv.get("from") == "prediction":
            prediction = conv.get("value")

    # if pred is empty, the score is 0
    """
    We occasionally encounter empty predictions from LLMs, mightbe a kind of content filtering policies. We tried with few of attmpts but still not resolved. 
    
    For example, we got this answer from o4-mini:
    * Error code: 400 - {'error': {'message': "The response was filtered due to the prompt triggering Azure OpenAI's content management policy. Please modify your prompt and retry. To learn more about our content filtering policies please read our documentation: \r\nhttps://go.microsoft.com/fwlink/?linkid=2198766. (request id: 20250721210031521237580bgOxZOpE) (request id: 20250721210028328166937C7pSa07Q) (request id: 20250721210027632501472a6M3oqe4)", 'type': '', 'param': 'prompt', 'code': 'content_filter'}}. Skipping.

    So we need to handle these corner cases to prevent potential misjudgement.
    """
    if cleaned_pred is not None:
        prediction = cleaned_pred.strip()
    elif prediction is not None:
        prediction = prediction.strip()
    else:
        print('[Warning] LLM prediction is empty, skipping evaluation for this item.')
        return 0.0

    # lowercase all words for case-insensitive comparison
    pred_normalized = normalize_text(prediction)
    gt_normalized = normalize_text(ground_truth)

    # check if the prediction contains the complete GT
    if gt_normalized in pred_normalized:
        return 1.0

    # split ground truth into pieces of words
    gt_words = gt_normalized.split()
    total_words = len(gt_words)
    
    # to find the longest matching substring in pred_normalized   
    for length in range(total_words - 1, 0, -1):
        for i in range(total_words - length + 1):
            sub_phrase_list = gt_words[i : i + length]
            sub_phrase = " ".join(sub_phrase_list)
            
            if sub_phrase in pred_normalized:
                return length / total_words

    return 0.0


def eval_rec_task(item, cleaned_pred=None):
    """ Evaluate the spatial understanding task by calculating the Intersection over Union (IoU) of bounding boxes.
    * A strict mode is used, meaning that the prediction must contain only four digits.
    """
    ground_truth = None
    prediction = None

    # parse conversations content
    for i, conv in enumerate(item.get("conversations", [])):
        if conv.get("from") == "gpt":
            ground_truth = conv.get("value")
        elif conv.get("from") == "prediction":
            prediction = conv.get("value")

    # if pred is empty, the score is 0
    if cleaned_pred is not None:
        prediction = cleaned_pred.strip()
    elif prediction is not None:
        prediction = prediction.strip()
    else:
        print('[Warning] LLM prediction is empty, skipping evaluation for this item.')
        return 0.0

    try:
        ans_bbox = list(map(int, re.findall(r'(\d+)', ground_truth)))
        pred_numbers = list(map(int, re.findall(r'(\d+)', prediction)))
        
        # Check if it is 4 values, if not the score is 0
        if len(pred_numbers) != 4:
            return 0.0

        # calculate iou
        iou = compute_iou(pred_numbers, ans_bbox)
        return iou

    except Exception:
        return 0.0


def evaluate_by_task(task_id, item, gpt_oss_resources, data_type):
    prediction_text = None
    cleaned_pred = None

    # For reasoning tasks
    if data_type == 'reasoning':
        for conv in item.get("conversations", []):
            if conv.get("from") == "prediction":
                prediction_text = conv.get("value")
                break

        if prediction_text is not None:
            match = re.search(r'<answer>\s*(.*?)\s*</answer>', prediction_text, re.DOTALL | re.IGNORECASE)
            if match:
                cleaned_pred = match.group(1).strip()
            else:
                cleaned_pred = prediction_text.strip()
        else:
            cleaned_pred = ""
    else:
        cleaned_pred = None

    if task_id in [11, 12]:
        """
        ---------------------------------
        Open Question Answering
        ---------------------------------

        * Tasks
            * #11 Lesion Diagnosis (open-set mode)
            * #12 Referring Expression Comprehension
        * Evaluation Method
            * Inspired by LLaVA (https://github.com/haotian-liu/LLaVA/blob/main/llava/eval/m4c_evaluator.py#L221) & ROUGE-L (https://huggingface.co/spaces/evaluate-metric/rouge)
            * First, a complete answer match is performed.  If the ground truth (GT) is completely present in the prediction, a score of 1.0 is awarded. 
            * Otherwise, all substrings of the GT are searched for within the Pred. The ratio of the number of words in the longest contious matching substring to the total number of words in the GT is used as the final score.
            * Finally, we calculate: score = corrected words/total answer words
        * Samples: 
            * GT="low grade adenoma", Pred="grade adenoma", Score=2/3=0.667
            * GT="low grade adenoma", Pred="low other_words adenoma", Score=1/3
            * GT="low grade adenoma", Pred="other_words", Score=0
        """
        return eval_open_vqa(item, cleaned_pred=cleaned_pred)

    elif task_id in [3, 7, 8, 9]:
        """
        ---------------------------------
        Yes-or-No Questions
        ---------------------------------
        * Tasks
            * #3 Rectum Retroflexion Identification
            * #7 Instrument Recognition
            * #8 Bleeding Warning
            * #9 Lesion Diagnosis (Y/N)
        * Evaluation Method
            * Searches for an independent 'yes' or 'no' (case-insensitive) in the Pred. If a word matching the GT is found, a score of 1.0 is awarded; otherwise, 0.0.
            * If 'yes' and 'no' are both present in the Pred, we use llm-as-a-judge to determine the only option for the score calculation.
        * Sample
            * GT="Yes", Pred="I think the answer is yes.", Score=1.0
            * GT="Yes", Pred="Yes, the image appears to show a low-grade adenoma area within the gastrointestinal tract. The adenoma is characterized by a small, round, reddish mass with a smooth surface, which is typical of low-grade adenomas. The surrounding tissue looks normal, and there is no evidence of high-grade dysplasia or invasive cancer", llm-as-a-judge=“yes”, Score=1.0
        """
        return eval_yes_or_no(item, gpt_oss_resources, cleaned_pred=cleaned_pred)

    elif task_id in [1, 4, 5, 6, 10, 14, 15, 17, "A", "B", "C", "D"]:
        """
        ---------------------------------
        Multiple Choice Questions
        ---------------------------------
        * Tasks
            * #1 Bowel Cleanliness Grading
            * #4 Intervention Stage Recognition
            * #5 Imaging quality assessment
            * #6 Imaging Modality Classification
            * #10 Lesion Diagnosis (single choice)    
            * #14 Early Cancer Grading (NICE criteria)
            * #15 Early Cancer Grading (PARIS criteria)
            * #17 Ulcerative Colitis Grading
        * Evaluation Method
            * Combines exact and partial matching. 
            * First, an exact match is attempted. If successful, a score of 1.0 is awarded. 
            * Otherwise, option numbers, option text, and option aliases (if available) are checked one by one. A match on any of these yields a score of 1.0. Otherwise, the score is 0.0.
            * Specifically, if more than one option is present in the Pred, we use llm-as-a-judge to determine the only option for the score calculation.
        * Sample 
            * GT="<B> Narrow Band Imaging (NBI)", Pred="Narrow Band Imaging is used.", Score=1.0
            * GT="<B> Narrow Band Imaging (NBI)", Pred="NBI is used.", Score=1.0
            * GT="<A> Normal", Pred="The result is <A> Normal", Score=1.0
            * GT="<A> Normal", Pred="The right answer is A.", Score=1.0
        """
        return eval_single_choice_vqa(item, gpt_oss_resources, cleaned_pred=cleaned_pred)

    elif task_id == 2:
        """
        ---------------------------------
        Two-Step Judgment Questions (True/False followed by justification) 
        ---------------------------------
        * Tasks
            * #2 Colonoscopy Completion Landmark Identification
        * Evaluation Method
            * The Pred's decision is compared to the GT. If consistent, a score of 0.5 is awarded; otherwise, 0.0. 
            * If the stance is correct, keyword matching is performed.  A match adds another 0.5 to the score.
        * Sample
            * GT="No. This image does not show any anatomical landmarks.", Pred="Maybe the image does not show any landmarks.", Score=1.0
            * GT="Yes. This image shows cecum.", Pred="yes", Score=0.5
            * GT="Yes. This image shows ileocecal valve.", Pred=“No. This image shows ileocecal valve.", Score=0.0
        """
        return eval_anatomical_landmark(item, cleaned_pred=cleaned_pred)

    elif task_id == 16:
        """
        ---------------------------------
        Numerical Range Questions
        ---------------------------------
        * Tasks
            * #16 Polyp Sizing
        * Evaluation Method 
            * First, a complete match is attempted. If the Pred contains the option number or answer text, a score of 1.0 is awarded. 
            * Otherwise, the GT range is broken into individual conditions. A score of 1.0 is awarded only if the Pred's natural language description satisfies all of these conditions.
            * If none of the strategies succeed, the score is 0.0.
            * In addition, if more than one option is present in the Pred, we use llm-as-a-judge to determine the only option for the score calculation.
        * Sample
            * GT="<C> 0mm < polyp < 6mm", Pred="The polyp is larger than 0 and smaller than 6mm.", Score=1.0
            * GT="<B> polyp ≥ 30mm", Pred="B", Score=1.0
            * GT="<C> 0mm < polyp < 6mm", Pred="0mm < polyp", Score=0.0 

        """
        return eval_numerical_range(item, gpt_oss_resources, cleaned_pred=cleaned_pred)

    elif task_id == 13: 
        """
        ---------------------------------
        Spatial understanding
        ---------------------------------
        * Tasks
            * #13 Referring Expression Comprehension
        * Evaluation Method
            * Coordinate values are extracted from the Pred and GT. The Intersection over Union (IoU) of the two bounding boxes (Bboxes) is calculated and used as the final score.
        * Sample
            * GT="{<624><606><967><1013>}", Pred="[624, 606, 967, 1013]", Score=1.0
        """
        return eval_rec_task(item, cleaned_pred=cleaned_pred)

    else:
        print(f"[Warning] task {task_id} is undefined!")
        return 0.0


def evaluator(args):
    gpt_oss_resources = None
    try:
        print("[Info] Loading gpt-oss model and tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(args.gpt_oss_path)
        model = AutoModelForCausalLM.from_pretrained(
            args.gpt_oss_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            low_cpu_mem_usage=True
        )
        gpt_oss_resources = {"tokenizer": tokenizer, "model": model, "device": "cuda"}
        print(f"[Info] gpt-oss model loaded successfully across {torch.cuda.device_count()} GPUs.")
    except Exception as e:
        print(f"[Warning] Failed to load gpt-oss model: {e}")
        return

    raw_task_id = args.task_id
    if raw_task_id.isdigit():
        task_id_to_run = int(raw_task_id)
    else:
        task_id_to_run = raw_task_id
    results_by_task: Dict[Union[int, str], List[float]] = {task_id_to_run: []}

    # get expected counts based on evaluation mode
    if args.eval_mode == 'pilot':
        counts_to_use = EXPECTED_COUNTS_PILOT
    elif args.eval_mode == 'all':
        counts_to_use = EXPECTED_COUNTS
    elif args.eval_mode == 'pert':
        counts_to_use = EXPECTED_COUNTS_PERT
    else:
        print("[Warning] Undefined evaluation mode, check your spelling.")

    # load the prediction json file
    with open(args.input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    actual_count = len(data)
    expected_count = counts_to_use.get(task_id_to_run)
    # Check if the actual count matches the expected count
    if expected_count is not None and actual_count != expected_count:
        raise ValueError(f"The number of samples for task {task_id_to_run} does not match! "
                         f"expect: {expected_count}, actual:{actual_count})")

    for item in tqdm(data):
        score = evaluate_by_task(task_id_to_run, item, gpt_oss_resources, args.data_type)
        results_by_task[task_id_to_run].append(score)

    # The reporting logic remains the same
    report = {"eval_results_by_task": {}}
    for task_id, scores in results_by_task.items():
        if scores:
            avg_score = (sum(scores) / len(scores))
            report["eval_results_by_task"][f"Task_{task_id}"] = {
                "average_score": round(avg_score, 8)
            }

    # save the score text file
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=4, ensure_ascii=False)
    
    # save the modified json file
    try:
        with open(args.input_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
    except Exception as e:
        print(f"[Warning] Failed to write modifications back to source file: {e}")


def main():
    # Create argument parser with raw text formatting to preserve help text formatting
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

    # Task ID to evaluate (MUT#1 ~ MUT#17)
    parser.add_argument("--task_id", type=str)
    # Input prediction files
    parser.add_argument("--input_file", type=str)
    # Output file path for evaluation results
    parser.add_argument("--output_file", type=str)
    # Evaluation mode: 'all' for full dataset, 'pilot' for pilot study subset (5000 samples)
    parser.add_argument("--eval_mode", type=str, choices=['all', 'pilot', 'pert'])
    parser.add_argument(
        "--data_type", 
        type=str, 
        choices=['reasoning', 'default'], 
        default='default', 
        help="Specify the data type. If 'reasoning', extracts answer from <answer> tags."
    )   
    parser.add_argument("--gpt_oss_path",type=str, default='cache/download-weights/gpt-oss-20b')

    args = parser.parse_args()

    # Run the evaluation engine with parsed arguments
    evaluator(args)


if __name__ == "__main__":
    main()