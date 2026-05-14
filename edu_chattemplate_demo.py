'''
this file applies the mistral style chat template to eduFeedback dataset
'''

from transformers import AutoTokenizer
import json
from datasets import Dataset

# load Tokenizer from the hub
model_id = "cognitivecomputations/dolphin-2.1-mistral-7b"
tokenizer = AutoTokenizer.from_pretrained(model_id)

# sys message used if there is no system message at the beginning of the conversation
DEFAULT_SYSTEM_MESSAGE = "You are Dolphin, a helpful AI tutor."


def apply_chat_template(example, tokenizer, default_system_message=DEFAULT_SYSTEM_MESSAGE):
    """Apply chat template using tokenizer"""
    system_prompt = [{"role": "system", "content": default_system_message}]
    
    # message formatting
    prompt_messages = system_prompt + [{"role": "user", "content": example["prompt"]}]
    chosen_messages = [{"role": "assistant", "content": example["chosen"]}]
    rejected_messages = [{"role": "assistant", "content": example["rejected"]}]

    # apply chat template
    return {
        "prompt": tokenizer.apply_chat_template(prompt_messages, tokenize=False),
        "chosen": tokenizer.apply_chat_template(chosen_messages, tokenize=False),
        "rejected": tokenizer.apply_chat_template(rejected_messages, tokenize=False),
    }


input_file = "/home/miria/cvxdpo/datasets/pref_dataset_alternate_edu_FINAL.json"
output_file = "test_edu_dataset.json"
print(len(output_file))

with open(input_file, "r", encoding="utf-8") as f:
    dataset = json.load(f)


# convert to HF Dataset for easy processing
dataset = Dataset.from_list(dataset)

# select range here (13750) was og
dataset = dataset.shuffle().select(range(23750))
dataset = dataset.train_test_split(test_size=2750/13750)

# apply transformation
dataset = dataset.map(apply_chat_template, fn_kwargs={"tokenizer": tokenizer})

# Save processed dataset
#dataset.to_json(output_file)

# save datasets to disk
dataset["train"].to_json("train_edu_dataset_subset.json", orient="records")
dataset["test"].to_json("test_edu_dataset_subset.json", orient="records")

print(f"Chat template applied and saved to {output_file}")
