import torch

from transformers import AutoTokenizer

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from my_configuration_llama import LlamaConfig
from my_flash_modeling_llama import LlamaForCausalLM

from train import smart_tokenizer_and_embedding_resize


DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "</s>"
        

def load_model_tokenizer(model_name_or_path, model_max_position_embeddings, rope_scaling_factor, rope_scaling_type, use_cache=True):

    Config, CausalLM, Tokenizer = LlamaConfig, LlamaForCausalLM, AutoTokenizer
    config = Config.from_pretrained(model_name_or_path, torch_dtype=torch.bfloat16)
    scaled_max_position_embeddings=int(model_max_position_embeddings * rope_scaling_factor)

    if config.rope_scaling is None:
        if rope_scaling_type is not None:
            config.rope_scaling={"type": rope_scaling_type, "factor": rope_scaling_factor}
            config.max_position_embeddings=scaled_max_position_embeddings
            if rope_scaling_type == "yarn":
                config.rope_scaling["original_max_position_embeddings"] = model_max_position_embeddings

    config.use_cache=use_cache
    
    print(config)
    
    print(f"load model from {model_name_or_path}")
    model = CausalLM.from_pretrained(
        pretrained_model_name_or_path=model_name_or_path, 
        config=config, torch_dtype=torch.bfloat16)
    model.to("cuda")
    model.eval()

    print("load tokenizer")
    tokenizer = Tokenizer.from_pretrained(model_name_or_path, use_fast=True)

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

    return model, tokenizer
