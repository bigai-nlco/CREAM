import logging
import random
import torch
import torch.distributed
import transformers

import numpy as np
import scipy.special as sp

from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence
from transformers import Trainer
from datasets import load_from_disk, load_dataset

from my_configuration_llama import LlamaConfig
from my_flash_modeling_llama import LlamaForCausalLM


IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "</s>"


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="")
    model_type: Optional[str] = field(default="")


@dataclass
class DataArguments:
    train_data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    valid_data_path: str = field(default=None, metadata={"help": "Path to the training data."})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_position_embeddings: int = field(
        default=4096,
        metadata={"help": "Maximum position embeddings."},
    )
    inference_length: int = field(
        default=4096,
        metadata={"help": "Maximum position embeddings."},
    )
    rope_scaling_type: Optional[str] = field(default=None)
    rope_scaling_factor: float = field(default=1.0)
    task_name: str = field(default=None)
    sigma: float = field(default=3.0)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels, position_ids = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels", "position_ids"))
        input_ids = [torch.tensor(x) for x in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = [torch.tensor(x) for x in labels]
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        position_ids = [torch.tensor(x) for x in position_ids]
        position_ids = torch.nn.utils.rnn.pad_sequence(position_ids, batch_first=True, padding_value=0)
        return dict(
            input_ids=input_ids,
            labels=labels,
            position_ids=position_ids,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


def Full_Length(examples, **kwargs):

    tokenizer = kwargs['tokenizer']
    scaled_max_position_embeddings = kwargs['scaled_max_position_embeddings']

    inputs = examples["text"]
    raw_model_inputs = tokenizer(inputs, padding=False, truncation=True, max_length=scaled_max_position_embeddings)
    position_ids = [torch.arange(len(ids), dtype=torch.long) for ids in raw_model_inputs["input_ids"]]

    model_inputs = {"input_ids": raw_model_inputs["input_ids"], "position_ids": position_ids, "labels": raw_model_inputs["input_ids"]}

    return model_inputs


def RandPos(examples, **kwargs):

    tokenizer = kwargs['tokenizer']
    scaled_max_position_embeddings = kwargs['scaled_max_position_embeddings']
    model_max_position_embeddings = kwargs['model_max_position_embeddings']

    inputs = examples["text"]
    model_inputs = tokenizer(inputs, padding=False, truncation=True, max_length=model_max_position_embeddings)
    position_ids = [torch.arange(len(ids), dtype=torch.long) for ids in model_inputs["input_ids"]]

    for pos_ids in position_ids:
        len_pos_ids = len(pos_ids)

        tot_pos_list = list(range(scaled_max_position_embeddings))
        new_pos_list = random.sample(tot_pos_list, len_pos_ids)
        new_pos_list.sort()
        pos_ids[:] = torch.tensor(new_pos_list, dtype=torch.long)

    model_inputs["position_ids"] = position_ids
    model_inputs["labels"] = model_inputs["input_ids"]

    return model_inputs


def PoSE(examples, **kwargs):

    tokenizer = kwargs['tokenizer']
    scaled_max_position_embeddings = kwargs['scaled_max_position_embeddings']
    model_max_position_embeddings = kwargs['model_max_position_embeddings']

    inputs = examples["text"]
    factor = scaled_max_position_embeddings // model_max_position_embeddings
    raw_model_inputs = tokenizer(inputs, padding=False, truncation=True, max_length=model_max_position_embeddings * factor)

    input_ids = []
    position_ids = []

    for ids in raw_model_inputs["input_ids"]:

        len_chunk = min(len(ids), model_max_position_embeddings)
        len_input = len(ids)
        lt1 = 0
        rt1 = random.randint(1, (len_chunk+1)//2)
        rt2 = random.randint(lt1+len_chunk, len_input)
        lt2 = rt2 - (len_chunk - (rt1-lt1))
        chunked_ids = ids[lt1:rt1] + ids[lt2:rt2]
        input_ids.append(chunked_ids)

        pos_ids = torch.arange(len(chunked_ids), dtype=torch.long)
        len_pos_ids = len(pos_ids)
        lt = 0
        rt = random.randint(lt, scaled_max_position_embeddings-len_pos_ids)

        pos_ids[:rt1-lt1] += lt
        pos_ids[rt1-lt1:] += rt
        position_ids.append(pos_ids)
    
    model_inputs = {"input_ids": input_ids, "position_ids": position_ids, "labels": input_ids}

    return model_inputs


def CREAM(examples, **kwargs):

    scaled_max_position_embeddings = kwargs['scaled_max_position_embeddings']
    model_max_position_embeddings = kwargs['model_max_position_embeddings']

    input_ids = []
    position_ids = []

    factor = scaled_max_position_embeddings // model_max_position_embeddings
    rand_num = random.randint(0, 1)
    if rand_num == 0:
        head_len = model_max_position_embeddings // 3
        tail_len = model_max_position_embeddings // 3
    else:
        head_len = 4 * factor
        tail_len = 4 * factor

    len_chunk = model_max_position_embeddings - head_len - tail_len
    pos_ids_1 = torch.arange(0, head_len, dtype=torch.long)
    pos_ids_3 = torch.arange(factor * model_max_position_embeddings - tail_len, factor * model_max_position_embeddings, dtype=torch.long)

    mu, sigma = 1 + factor, kwargs['sigma']
    x = np.linspace(2, factor * 2, 1000)
    cdf = 0.5 * (1 + sp.erf((x - mu) / (sigma * np.sqrt(2))))

    for ids in examples["input_ids"]:
        uniform_random = np.random.rand()
        rand_factor = np.interp(uniform_random, cdf, x).astype(int)

        assert rand_factor in list(np.arange(2, factor * 2 + 1))

        end_id =  random.randint(int(head_len + ((len_chunk - 1) * (rand_factor / 2))), ((rand_factor / 2) * model_max_position_embeddings) - tail_len - 1)

        pos_ids_2 = torch.arange(end_id - (len_chunk - 1), end_id + 1, dtype=torch.long)
        pos_ids = torch.cat([pos_ids_1, pos_ids_2, pos_ids_3]).tolist()
        position_ids.append(pos_ids)

        if end_id >= len(ids):
            input_pos_ids_2 = torch.arange(len(ids) - tail_len - (len_chunk - 1) - 1, len(ids) - tail_len, dtype=torch.long)
        else:
            input_pos_ids_2 = pos_ids_2
        input_pos_ids_3 = torch.arange(len(ids) - tail_len, len(ids), dtype=torch.long)

        input_pos_ids = torch.cat([pos_ids_1, input_pos_ids_2, input_pos_ids_3])
        inp_ids = torch.tensor(ids)[input_pos_ids].tolist()
        input_ids.append(inp_ids)

    model_inputs = {"input_ids": input_ids, "position_ids": position_ids, "labels": input_ids}

    return model_inputs


def test_preprocess_function(examples, tokenizer, inference_length):

    inputs = examples["text"]
    model_inputs = tokenizer(inputs, padding=False, truncation=True, max_length=inference_length)
    position_ids = [torch.arange(len(ids), dtype=torch.long) for ids in model_inputs["input_ids"]]
    model_inputs["position_ids"] = position_ids
    model_inputs["labels"] = model_inputs["input_ids"]

    return model_inputs


def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    config = LlamaConfig.from_pretrained(model_args.model_name_or_path, cache_dir=training_args.cache_dir)
    scaled_max_position_embeddings=int(training_args.model_max_position_embeddings * training_args.rope_scaling_factor)
    config.max_position_embeddings=scaled_max_position_embeddings

    if training_args.rope_scaling_type is not None:
        config.rope_scaling={"type": training_args.rope_scaling_type, "factor": training_args.rope_scaling_factor}
        if "yarn" in training_args.rope_scaling_type:
            config.rope_scaling["original_max_position_embeddings"] = training_args.model_max_position_embeddings

    model = LlamaForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        config=config,
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        padding_side="right",
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
            tokenizer=tokenizer,
            model=model,
        )
    tokenizer.add_special_tokens(
        {
            "eos_token": DEFAULT_EOS_TOKEN,
            "bos_token": DEFAULT_BOS_TOKEN,
            "unk_token": DEFAULT_UNK_TOKEN,
        }
    )

    raw_train_datasets = load_from_disk(data_args.train_data_path)

    if training_args.local_rank > 0: 
        torch.distributed.barrier()

    num_proc = 16
    if 'randPos' in training_args.task_name:
        print('=' * 50)
        print('>> randPos Process')
        print('=' * 50)
        process_func = RandPos
    elif 'pose' in training_args.task_name:
        print('=' * 50)
        print('>> PoSE Process')
        print('=' * 50)
        process_func = PoSE
    else:
        print('=' * 50)
        print('>> CREAM Process')
        print(f'>> sigma: {training_args.sigma}')
        print('=' * 50)
        process_func = CREAM
        num_proc = 1

    train_dataset = raw_train_datasets.map(
        process_func,
        batched=True,
        batch_size=3000,
        num_proc=num_proc,
        remove_columns=raw_train_datasets.column_names,
        load_from_cache_file=True,
        desc="Data Process on train dataset",
        fn_kwargs={
            "sigma": training_args.sigma,
            "tokenizer": tokenizer, 
            "scaled_max_position_embeddings": scaled_max_position_embeddings, 
            "model_max_position_embeddings": training_args.model_max_position_embeddings
        }
    )
    
    print(train_dataset)

    raw_valid_datasets = load_from_disk(data_args.valid_data_path)

    valid_dataset = raw_valid_datasets.map(
        test_preprocess_function,
        batched=True,
        batch_size=3000,
        num_proc=1,
        remove_columns=raw_valid_datasets.column_names,
        load_from_cache_file=True, # not args.overwrite_cache
        desc="Data Process on valid dataset",
        fn_kwargs={"tokenizer": tokenizer, "inference_length": training_args.inference_length}
    )

    if training_args.local_rank == 0:
        torch.distributed.barrier()
    
    if training_args.local_rank == 0:
        print(len(train_dataset))
        for index in random.sample(range(len(train_dataset)), 3):
            print(f"Sample {index} of the training set: {train_dataset[index]}.")
    
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    data_module = dict(train_dataset=train_dataset, eval_dataset=valid_dataset, data_collator=data_collator)

    #Tell Trainer not to attempt DataParallel
    model.is_parallelizable = True
    model.model_parallel = True

    trainer = Trainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
    model.config.use_cache = False

    if training_args.do_train:
        logging.info("*** Start Training ***")
        trainer.train()
        trainer.save_state()
        safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)

    if training_args.do_eval:
        logging.info("*** Evaluate on valid set***")
        metrics = trainer.evaluate(eval_dataset=valid_dataset)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


if __name__ == "__main__":
    train()
