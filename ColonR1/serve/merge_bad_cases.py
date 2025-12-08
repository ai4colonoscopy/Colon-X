import json
import argparse
from typing import List, Dict, Any, Tuple
import os
import re

PATH_PREFIX_MARKER = "Positive-images/"

PROMPT_NOTE_TEMPLATE = " Note: Avoid the following incorrect answers and provide a different response. Historical incorrect answers: {}"

def to_relative_path(absolute_path: str) -> str:
    try:
        index = absolute_path.rfind(PATH_PREFIX_MARKER)
        if index != -1:
            return absolute_path[index:]
        else:
            return absolute_path
    except Exception as e:
        print(f"Error: An error occurred while converting path '{absolute_path}': {e}")
        return absolute_path

def create_bad_case_lookup_from_data(bad_case_data: List[Dict[str, Any]]) -> Dict[Tuple[str, str], str]:
    print(f"Creating lookup map from {len(bad_case_data)} raw bad case items...")
    bad_case_problem_lookup = {}
    processed_keys = 0
    skipped_keys = 0
    answer_extraction_count = 0

    for case in bad_case_data:
        if not isinstance(case, dict):
            print("Warning: Skipping non-dictionary item in bad case data.")
            skipped_keys += 1
            continue

        img_path_orig = case.get('image')
        solution = case.get('solution')
        original_problem_text = case.get('problem', "")
        wrong_predictions = case.get("wrong_pred")

        if not (isinstance(img_path_orig, str) and solution is not None):
            print(f"Warning: Skipping bad case item due to missing 'image' or 'solution'.")
            skipped_keys += 1
            continue

        relative_image_path = to_relative_path(img_path_orig)
        
        new_problem_to_store = original_problem_text

        if isinstance(wrong_predictions, list) and wrong_predictions:
            all_extracted_answer_tags = []
            for pred_string in wrong_predictions:
                if isinstance(pred_string, str):
                    found_tags = re.findall(r'<answer>.*?</answer>', pred_string, re.DOTALL | re.IGNORECASE)
                    if found_tags:
                        all_extracted_answer_tags.extend(found_tags)
            
            if all_extracted_answer_tags:
                answer_extraction_count += len(all_extracted_answer_tags) 

                unique_answer_tags = list(dict.fromkeys(all_extracted_answer_tags))
                
                wrong_answers_str = ", ".join(unique_answer_tags)
                note_to_add = PROMPT_NOTE_TEMPLATE.format(wrong_answers_str)

                if PROMPT_NOTE_TEMPLATE.split('{}')[0] not in original_problem_text:
                    new_problem_to_store = f"{original_problem_text}{note_to_add}"
                else:
                    problem_base = original_problem_text.split(PROMPT_NOTE_TEMPLATE.split('{}')[0])[0]
                    new_problem_to_store = f"{problem_base.strip()}{note_to_add}"


        key = (relative_image_path.strip(), str(solution).strip())
        bad_case_problem_lookup[key] = new_problem_to_store
        processed_keys += 1

    print(f"Created lookup map with {processed_keys} entries (skipped {skipped_keys} invalid entries).")
    print(f"Extracted {answer_extraction_count} <answer> tags in total to update 'problem' fields (duplicates removed before writing).")
    return bad_case_problem_lookup


def update_original_data(original_data: List[Dict[str, Any]],
                             bad_case_problem_lookup: Dict[Tuple[str, str], str]):

    output_data = []
    replaced_problem_count = 0
    kept_original_count = 0
    path_conversion_failures_original = 0

    for index, item_original in enumerate(original_data):
        if not isinstance(item_original, dict):
            print(f"Warning: Original data item at index {index} is not a dictionary, skipping.")
            continue

        item_to_add = item_original.copy() 

        original_image_path = item_original.get("image")
        original_solution = item_original.get("solution")

        if isinstance(original_image_path, str) and original_solution is not None:
            relative_image_path = to_relative_path(original_image_path)
            
            if relative_image_path == original_image_path and PATH_PREFIX_MARKER in original_image_path:
                 path_conversion_failures_original += 1

            item_to_add['image'] = relative_image_path

            key_original_lookup = (relative_image_path.strip(), str(original_solution).strip())

            if key_original_lookup in bad_case_problem_lookup:
                item_to_add['problem'] = bad_case_problem_lookup[key_original_lookup]
                replaced_problem_count += 1
            else:
                kept_original_count += 1
        else:
            kept_original_count += 1

        output_data.append(item_to_add)

    if path_conversion_failures_original > 0:
        print(f"  Warning: During processing of original data, {path_conversion_failures_original} image paths might not have been converted to relative paths successfully.")

    return output_data

def main():
    parser = argparse.ArgumentParser(description="Step 1: Read bad_case file. Step 2: Read original file. Step 3: Merge files into a new output file, applying bad case modifications in memory. NO input files will be modified.")
    parser.add_argument("--bad_case_file", required=True, type=str, help="Path to the bad cases JSON file (will be READ only).")
    parser.add_argument("--original_file", required=True, type=str, help="Path to the original JSON data file (will be READ only).")
    parser.add_argument("--output_file", required=True, type=str, help="Path for the merged output JSON file.")
    args = parser.parse_args()

    if not os.path.exists(args.bad_case_file):
        print(f"Error: Bad case file not found at '{args.bad_case_file}'. Cannot proceed.")
        return
    try:
        with open(args.bad_case_file, 'r', encoding='utf-8') as f:
            raw_bad_case_data = json.load(f)
        print(f"Successfully read bad case file: '{args.bad_case_file}' ({len(raw_bad_case_data)} items)")
    except Exception as e:
        print(f"Error: Failed to read or parse JSON from bad case file '{args.bad_case_file}': {e}")
        return

    if not os.path.exists(args.original_file):
        print(f"Error: Original file not found at '{args.original_file}'. Cannot proceed.")
        return
    try:
        with open(args.original_file, 'r', encoding='utf-8') as f:
            original_data = json.load(f)
    except Exception as e:
        print(f"Error: Failed to read or parse JSON from original file '{args.original_file}': {e}")
        return


    bad_case_lookup_map = create_bad_case_lookup_from_data(raw_bad_case_data)

    merged_data = update_original_data(original_data, bad_case_lookup_map)

    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    try:
        with open(args.output_file, 'w', encoding='utf-8') as f:
            json.dump(merged_data, f, indent=4, ensure_ascii=False)
        print(f"Successfully wrote merged data to '{args.output_file}'")
    except Exception as e:
        print(f"Error: Failed to write output file '{args.output_file}': {e}")

    print('All done!!')

if __name__ == "__main__":
    main()