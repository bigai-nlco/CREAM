import argparse
import json
import logging
import pathlib
import random
import sys
import torch

from copy import deepcopy
from tqdm import tqdm
from xopen import xopen
from statistics import mean

from utils import get_kv_retrieval_prompt

import sys
sys.path.append('/home/user/CREAM/src/evaluation')
from base_utils import load_model_tokenizer


logger = logging.getLogger(__name__)
random.seed(0)


def main(
    input_path,
    model_name_or_path,
    task_name,
    gold_index,
    max_prompt_length,
    model_max_position_embeddings,
    rope_scaling_factor,
    rope_scaling_type,
    max_new_tokens,
    output_path,
    query_aware_contextualization=False,
    use_random_ordering=False,
    prompt_mention_random_ordering=False,
):
    # Create directory for output path if it doesn"t exist.
    pathlib.Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        raise ValueError("Unable to find CUDA device with torch. Please use a CUDA device to run this script.")

    model, tokenizer = load_model_tokenizer(
        model_name_or_path, model_max_position_embeddings, rope_scaling_factor, rope_scaling_type)
    tokenizer.padding_side = "left"


    examples = []
    prompts = []
    prompt_lengths = []

    # Fetch all of the prompts
    if task_name == 'kv':
        with xopen(input_path) as fin:
            for line in tqdm(fin):
                input_example = json.loads(line)
                # Get the prediction for the input example
                ordered_kv_records = deepcopy(input_example["ordered_kv_records"])
                key = input_example["key"]
                value = input_example["value"]

                original_kv_index = ordered_kv_records.index([key, value])
                # Remove the kv from its original index
                original_kv = ordered_kv_records.pop(original_kv_index)
                ordered_kv_records.insert(gold_index, original_kv)

                kv_prompt = get_kv_retrieval_prompt(
                    data=ordered_kv_records, key=key, query_aware_contextualization=query_aware_contextualization
                )

                prompt_length = len(tokenizer(kv_prompt)["input_ids"])
                if max_prompt_length < prompt_length:
                    logger.info(
                        f"Skipping prompt with length {prompt_length}, which "
                        f"is greater than maximum prompt length {max_prompt_length}"
                    )
                    continue

                prompts.append(kv_prompt)
                examples.append(deepcopy(input_example))
                prompt_lengths.append(prompt_length)

    print(f'Average Length: {mean(prompt_lengths)}.')


    responses = []
    # for batched_prompts in tqdm(chunks(prompts, batch_size), total=math.ceil(len(prompts) / batch_size)):
    for prompt in tqdm(prompts):
        # inputs = tokenizer(batched_prompts, return_tensors="pt", padding=True).to(model.device)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            use_cache=True,
        )
        for i, generated_sequence in enumerate(outputs):
            input_ids = inputs["input_ids"][i]
            text = tokenizer.decode(generated_sequence, skip_special_tokens=True, clean_up_tokenization_spaces=True)

            if input_ids is None:
                prompt_length = 0
            else:
                prompt_length = len(
                    tokenizer.decode(
                        input_ids,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=True,
                    )
                )
            new_text = text[prompt_length:]
            responses.append(new_text)

    with open(output_path, "w") as f:
        for example, response in zip(examples, responses):
            output_example = {}
            output_example["model"] = model_name_or_path
            output_example["model_answer"] = response
            if task_name == 'kv':
                output_example["golden"] = example['value']
            elif task_name == 'qa':
                output_example["model_prompt_mention_random_ordering"] = prompt_mention_random_ordering
                output_example["model_use_random_ordering"] = use_random_ordering
                output_example["golden"] = example['answers']
            f.write(json.dumps(output_example, indent=4) + "\n")


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s - %(module)s - %(levelname)s - %(message)s", level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", help="Path to data with questions and documents to use.", required=True)
    parser.add_argument("--model_name_or_path", help="Model to use in generating responses", required=True)
    parser.add_argument("--task_name",  type=str, required=True)
    parser.add_argument("--gold_index", help="Move the key to retrieve to this index", type=int, required=True)
    parser.add_argument("--max_prompt_length", type=int, default=4096)
    parser.add_argument("--model_max_position_embeddings", type=int, required=True)
    parser.add_argument("--rope_scaling_factor", type=float, default=1.0)
    parser.add_argument("--rope_scaling_type", type=str, default=None)
    parser.add_argument("--max_new_tokens", help="Maximum number of new tokens to generate", type=int, default=100)
    parser.add_argument("--output_path", help="Path to write output file of generated responses", required=True)

    parser.add_argument(
        "--query_aware_contextualization",
        action="store_true",
        help="Place the question both before and after the documents.",
    )
    parser.add_argument(
        "--use_random_ordering",
        action="store_true",
        help="Randomize the ordering of the distractors, rather than sorting by relevance.",
    )
    parser.add_argument(
        "--prompt_mention_random_ordering",
        action="store_true",
        help="Mention that search results are ordered randomly in the prompt",
    )
    
    args = parser.parse_args()

    logger.info("running %s", " ".join(sys.argv))
    main(
        args.input_path,
        args.model_name_or_path,
        args.task_name,
        args.gold_index,
        args.max_prompt_length,
        args.model_max_position_embeddings,
        args.rope_scaling_factor,
        args.rope_scaling_type,
        args.max_new_tokens,
        args.output_path,
        args.query_aware_contextualization,
        args.use_random_ordering,
        args.prompt_mention_random_ordering,
    )
    logger.info("finished running %s", sys.argv[0])
