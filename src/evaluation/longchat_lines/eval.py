import os
import re
import random
import argparse
import torch
import torch.distributed
import datasets

from tqdm import tqdm
from fastchat.model import get_conversation_template

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from base_utils import load_model_tokenizer


gpu_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def truncate(tokenizer, prompt, max_length):
    split_data = prompt.split('\n')

    pattern = r'in line (\w+-\w+)\?'
    match = re.search(pattern, split_data[-1])
    if match:
        extracted_string = match.group(1)
    else:
        print(prompt)
        return prompt

    input = tokenizer(prompt, return_tensors="pt")
    prompt_length = input.input_ids.shape[-1]

    while prompt_length > max_length:
        
        while True:
            ids = random.sample(range(1, len(split_data) - 1), 1)[0]
            if extracted_string in split_data[ids]:
                continue
            del split_data[ids]
            break
        
        input = tokenizer('\n'.join(split_data), return_tensors="pt")
        prompt_length = input.input_ids.shape[-1]

    return '\n'.join(split_data)


def test_lines_one_sample(model, tokenizer, test_case):
    prompt = test_case["prompt"]
    expected_number = test_case["expected_number"]

    conv = get_conversation_template("vicuna")
    print(f"Using conversation template: {conv.name}")

    conv.append_message(conv.roles[0], prompt)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input = tokenizer(prompt, return_tensors="pt")
    prompt_length = input.input_ids.shape[-1]

    output = model.generate(
        input_ids=input.input_ids.to(model.device), 
        min_new_tokens=5, 
        max_new_tokens=35, 
        use_cache=True,
        pad_token_id=tokenizer.eos_token_id
    )[0]
    output = output[prompt_length:]
    output = tokenizer.batch_decode([output], skip_special_tokens=True)[0]

    # Matching the first digit of the model output.
    response_number = re.findall("\d+", output)
    if response_number is not None and len(response_number) > 0:
        response_number = int(response_number[0])
    else:
        print(f"Got unparsable result")
        response_number = -1

    summary = f"Label: {expected_number}, Predict: {output}, Parsed: {response_number}, prompt length: {prompt_length}".replace('\n', ' ')
    print(summary)

    return expected_number == response_number, prompt_length, summary


def main():

    # add parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_shortest_only", action="store_true", default=False)
    parser.add_argument("--model_max_position_embeddings", type=int, default=4096)
    parser.add_argument("--rope_scaling_factor", type=float, default=1.0)
    parser.add_argument("--rope_scaling_type", type=str, default=None)
    parser.add_argument("--model_name", type=str, default="llama-7b")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--model_name_or_path", type=str, default=None)
    args = parser.parse_args()

    model, tokenizer = load_model_tokenizer(
        model_name_or_path=args.model_name_or_path, 
        model_max_position_embeddings=args.model_max_position_embeddings, 
        rope_scaling_factor=args.rope_scaling_factor, 
        rope_scaling_type=args.rope_scaling_type
    )

    lines_dataset = datasets.load_from_disk('/root/dataset/PoSE-Datasets/LongChat-Lines')
    lines = list(lines_dataset.keys())

    if args.eval_shortest_only:
        lines = [min(lines)]

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    output_file = os.path.join(args.output_dir, f"response_{args.model_name}.txt")

    for num_lines in lines:
        print(f"************ Start testing {num_lines} lines per LRT prompt ************")

        num_correct = 0
        avg_length = 0

        test_cases = lines_dataset[num_lines]
        for test_case in tqdm(test_cases):
            correct, prompt_length, _ = test_lines_one_sample(model=model, tokenizer=tokenizer, test_case=test_case)
            avg_length += prompt_length / len(test_cases)
            num_correct += correct
        accuracy = num_correct / len(test_cases)

        with open(output_file, "a+") as f:
            f.write(f"************ Finish testing {num_lines} lines per prompt with average prompt length {avg_length}, accuracy: {accuracy} ************\n\n")

        print(f"************ Finish testing {num_lines} lines per prompt with average prompt length {avg_length}, accuracy: {accuracy} ************")
        if args.eval_shortest_only:
            break


if __name__ == "__main__":
    main()
