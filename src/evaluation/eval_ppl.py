import os
import argparse
import torch
import torch.distributed

from datasets import load_from_disk
from tqdm import tqdm

from base_utils import load_model_tokenizer

gpu_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def compute_perplexity(
    encodings, model, tokenizer, add_start_token: bool = True, max_length=None, sliding_window_step=256, truncate=False
):
    r"""Compute "sliding window" perplexity on a dataset. Validated against the calculations reported in arXiv 2306.15595"""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if add_start_token:
        # leave room for <BOS> token to be added:
        assert (
            tokenizer.bos_token is not None
        ), "Input model must already have a BOS token if using add_start_token=True. Please use a different model, or set add_start_token=False"
        max_tokenized_len = max_length - 1
    else:
        max_tokenized_len = max_length

    encoded_texts = encodings["input_ids"]
    attn_masks = encodings["attention_mask"]

    encoded_texts = [x[0:max_tokenized_len] for x in encoded_texts]
    attn_masks = [x[0:max_tokenized_len] for x in attn_masks]

    if max_length and truncate:
        sliding_window_step = max_tokenized_len

    window_size = 32768

    pbar = tqdm(total=len(encoded_texts))
    nlls = []
    total_nll = torch.tensor(0,dtype=torch.float64).to(device)
    total_token_cnt = 0
    for encoding_index in range(0, len(encoded_texts)):

        labels = torch.tensor(encoded_texts[encoding_index:encoding_index+1])
        seq_len = labels.size(1)

        prev_end_loc = 0
        for begin_loc in range(0, seq_len, sliding_window_step):

            end_loc = min(begin_loc + window_size, seq_len)
            trg_len = end_loc - prev_end_loc
            input_ids = labels[:, begin_loc:end_loc].to(device)

            if add_start_token:
                bos_tokens_tensor = torch.tensor(
                    [[tokenizer.bos_token_id]] * input_ids.size(dim=0)).to(device)
                input_ids = torch.cat(
                    [bos_tokens_tensor, input_ids], dim=1)

            target_ids = input_ids.clone()

            with torch.no_grad():
                outputs = model(input_ids, labels=target_ids)
                neg_log_likelihood = outputs.loss
                total_nll += neg_log_likelihood * trg_len
                total_token_cnt += trg_len
            
            nlls.append(neg_log_likelihood)

            ppl = float(torch.exp(total_nll / total_token_cnt).float().cpu())
            pbar.set_postfix(ppl=ppl)

            prev_end_loc = end_loc
            if end_loc == seq_len:
                break

        pbar.update(1)

    ppl = float(torch.exp(total_nll / total_token_cnt).float().cpu())
    return {"mean_perplexity": ppl}


def main():

    # add parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--min_input_tokens", type=int, default=500)
    parser.add_argument("--max_input_tokens", type=int, default=1000)
    parser.add_argument("--step", type=int, default=500)
    parser.add_argument("--eval_nums", type=int, default=50)
    parser.add_argument("--sliding_window_step", type=int, default=256)
    parser.add_argument('--window_length_list', type=int, nargs='+', default=[])
    parser.add_argument("--truncate", action="store_true", default=False)
    parser.add_argument("--model_max_position_embeddings", type=int, default=4096)
    parser.add_argument("--rope_scaling_factor", type=float, default=1.0)
    parser.add_argument("--rope_scaling_type", type=str, default=None)
    parser.add_argument("--input_field", type=str, default="text")
    parser.add_argument("--model_name", type=str, default="llama2-7b")
    parser.add_argument("--model_name_or_path", type=str, default="")
    parser.add_argument("--dataset_name", type=str, default="scrolls-gov_report")
    parser.add_argument("--path_to_dataset", type=str, default="")
    parser.add_argument("--path_to_output_dir", type=str, default="results/ppls")
    args = parser.parse_args()

    model, tokenizer = load_model_tokenizer(
        model_name_or_path=args.model_name_or_path, 
        model_max_position_embeddings=args.model_max_position_embeddings, 
        rope_scaling_factor=args.rope_scaling_factor, 
        rope_scaling_type=args.rope_scaling_type
    )

    input_texts = load_from_disk(args.path_to_dataset)

    def tokenize(example):
        tokenized = tokenizer(
            example[args.input_field],
            add_special_tokens=False,
            padding=True,
            truncation=True,
            max_length=args.max_input_tokens - 1, # leave room for <BOS> token to be added
            return_attention_mask=True,
        )
        example["input_ids"] = tokenized["input_ids"]
        example["attention_mask"] = tokenized["attention_mask"]
        example["tokenized_len"] = len(tokenized["input_ids"])
        return example

    input_texts = input_texts.map(
        tokenize,
        num_proc=8,
        desc='tokenize'
    )

    if args.min_input_tokens:
        input_texts = input_texts.filter(
            lambda x: x["tokenized_len"] >= args.min_input_tokens - 1, num_proc=8)
    if args.eval_nums:
        input_texts = input_texts[:args.eval_nums]

    ppl_list = []
    context_window_size = args.window_length_list
    print(context_window_size)

    for ctx_size in context_window_size:
        # if args.truncate is True, we calucate the ppl on the whole input text
        # otherwise, we calucate the ppl with sliding window
        ppl = compute_perplexity(encodings=input_texts, model=model, tokenizer=tokenizer, add_start_token=True, max_length=ctx_size, sliding_window_step=args.sliding_window_step, truncate=args.truncate)["mean_perplexity"]

        print(f"model: {args.model_name}; context window size: {ctx_size}; ppl: {ppl}")
        
        ppl_list.append(ppl)

    if not os.path.exists(args.path_to_output_dir):
        os.makedirs(args.path_to_output_dir)

    path_to_output_fn = f"{args.path_to_output_dir}/{args.model_name}+{args.dataset_name}"
    with open(path_to_output_fn, "w") as f:
        f.write(f"model: {args.model_name}\n")
        f.write(f"length: {', '.join(map(str, context_window_size))}\n")
        f.write(f"ppl: {', '.join(map(str, ppl_list))}\n")


if __name__ == "__main__":
    main()
