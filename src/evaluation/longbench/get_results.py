import os
import torch
import json
import random
import argparse
import pickle

import numpy as np

from tqdm import tqdm
from datasets import load_from_disk

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from base_utils import load_model_tokenizer


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, default=None)
    parser.add_argument("--model_max_position_embeddings", type=int, default=4096)
    parser.add_argument("--rope_scaling_factor", type=float, default=1.0)
    parser.add_argument("--rope_scaling_type", type=str, default=None)
    parser.add_argument("--data_root_path", type=str, default='')
    parser.add_argument("--out_path_root", type=str, default=None)
    return parser.parse_args(args)


def preprocess_data(data, tokenizer, max_length, prompt_format, device):
    meta_list = []
    max_sample_len = 0
    for json_obj in tqdm(data, desc="preprocess"):
        prompt = prompt_format.format(**json_obj)

        # truncate to fit max_length (we suggest truncate in the middle, since the left and right side may contain crucial instructions)
        tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
        
        if len(tokenized_prompt) > max_length:
            half = int(max_length / 2)
            prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True) + tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)

        if dataset not in ["trec", "triviaqa", "samsum", "lcc", "repobench-p"]: # chat models are better off without build prompts on these tasks
            prompt = f"[INST]{prompt}[/INST]"

        model_input = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
        context_length = model_input.input_ids.shape[-1]
        max_sample_len = max(context_length, max_sample_len)

        meta_list.append({
            'input': model_input,
            'answers': json_obj['answers'], 
            'length': json_obj['length'], 
            'all_classes': json_obj['all_classes']
        })
    
    return meta_list, max_sample_len


def get_pred(meta_list, model, tokenizer, dataset, max_gen, device, out_path):

    model.to(device)

    result = []
    for meta in tqdm(meta_list, desc='generate'):
        input = meta['input']
        context_length = input.input_ids.shape[-1]
        if dataset == "samsum": # prevent illegal output on samsum (model endlessly repeat "\nDialogue"), might be a prompting issue
            output = model.generate(
                **input,
                max_new_tokens=max_gen,
                num_beams=1,
                do_sample=False,
                temperature=1.0,
                min_length=context_length+40,
                use_cache=True,
                eos_token_id=[tokenizer.eos_token_id, tokenizer.encode("\n", add_special_tokens=False)[-1]],
            )[0]
        else:
            output = model.generate(
                **input,
                max_new_tokens=max_gen,
                num_beams=1,
                do_sample=False,
                use_cache=True,
                temperature=1.0,
            )[0]
        pred = tokenizer.decode(output[context_length:], skip_special_tokens=True)

        result.append({
            "pred": pred, 
            "answers": meta["answers"], 
            "all_classes": meta["all_classes"], 
            "length": meta["length"]
        })
        
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4, ensure_ascii=False)
        f.write('\n')


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


if __name__ == '__main__':
    seed_everything(42)
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    datasets = ["narrativeqa", "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "musique", \
                "gov_report", "qmsum", "multi_news", "trec", "triviaqa", "samsum", "passage_count", "passage_retrieval_en", "lcc", "repobench-p"]

    # we design specific prompt format and max generation length for each task, feel free to modify them to optimize model output
    dataset2prompt = json.load(open("src/evaluation/longbench/dataset2prompt.json", "r"))
    dataset2maxlen = json.load(open("src/evaluation/longbench/dataset2maxlen.json", "r"))

    if not os.path.exists(args.out_path_root):
        os.makedirs(args.out_path_root)

    model, tokenizer = load_model_tokenizer(
        model_name_or_path=args.model_name_or_path, 
        model_max_position_embeddings=args.model_max_position_embeddings, 
        rope_scaling_factor=args.rope_scaling_factor, 
        rope_scaling_type=args.rope_scaling_type
    )

    if args.rope_scaling_type is None:
        max_seq_length = 3500
    else:
        max_seq_length = 31500
        
    print(max_seq_length)

    for dataset in datasets:
        print(dataset)
        data = load_from_disk(f'{args.data_root_path}/{dataset}')

        prompt_format = dataset2prompt[dataset]
        max_gen = dataset2maxlen[dataset]

        dataset_cache_path = f'/root/datasets/longbench_llama/{dataset}'
        if os.path.exists(dataset_cache_path):
            with open(dataset_cache_path, 'rb') as rp:
                meta_list = pickle.load(rp)
        else:
            meta_list, max_sample_len = preprocess_data(data, tokenizer, max_seq_length, prompt_format, device)
            with open(dataset_cache_path, 'wb') as wp:
                pickle.dump(meta_list, wp)

        out_path = os.path.join(args.out_path_root, f"{dataset}-{args.rope_scaling_type}.jsonl")

        try:
            get_pred(meta_list, model, tokenizer, dataset, max_gen, device, out_path)
        except Exception as e:
            print(dataset, max_sample_len)
            print(e)

        torch.cuda.empty_cache()
