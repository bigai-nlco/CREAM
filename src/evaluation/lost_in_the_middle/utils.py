import json
import pathlib

from typing import List, Tuple


DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "</s>"


def read_file(file_name):
    with open(file_name) as f:
        if 'txt' in file_name:
            json_data = f.read()
            json_data = "[" + json_data.replace("}\n{", "},\n{") + "]"
            all_examples = json.loads(json_data)

    return all_examples


PROMPTS_ROOT = (pathlib.Path(__file__).parent / "prompts").resolve()
def get_kv_retrieval_prompt(
    data: List[Tuple[str, str]],
    key: str,
    query_aware_contextualization: bool = False,
):
    if not data:
        raise ValueError(f"Provided `data` must be truthy, got: {data}")
    if not key:
        raise ValueError(f"Provided `key` must be truthy, got: {key}")
    if key not in [x[0] for x in data]:
        raise ValueError(f"Did not find provided `key` {key} in data {data}")
    if len(data) != len(set([x[0] for x in data])):
        raise ValueError(f"`data` has duplicate keys: {data}")
    if len(data) < 2:
        raise ValueError(f"Must have at least 2 items in data: {data}")

    if query_aware_contextualization:
        with open(PROMPTS_ROOT / "kv_retrieval_with_query_aware_contextualization.prompt") as f:
            prompt_template = f.read().rstrip("\n")
    else:
        with open(PROMPTS_ROOT / "kv_retrieval.prompt") as f:
            prompt_template = f.read().rstrip("\n")

    # Format the KV data into a string
    formatted_kv_records = ""
    for index, record in enumerate(data):
        start_character = "{" if index == 0 else " "
        data_string = f'"{record[0]}": "{record[1]}"'
        end_character = ",\n" if index != len(data) - 1 else "}"
        formatted_kv_records += start_character + data_string + end_character

    return prompt_template.format(formatted_kv_records=formatted_kv_records, key=key)
