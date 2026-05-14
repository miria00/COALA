

from transformers import AutoTokenizer
import json
from datasets import Dataset

# Load Tokenizer from the hub
model_id = "cognitivecomputations/dolphin-2.1-mistral-7b"  # Replace with your model id
tokenizer = AutoTokenizer.from_pretrained(model_id)

# System message used if there is no system message at the beginning of the conversation
DEFAULT_SYSTEM_MESSAGE = "You are Dolphin, a helpful AI tutor."


def apply_chat_template(example, tokenizer, default_system_message=DEFAULT_SYSTEM_MESSAGE):
    """Apply chat template using tokenizer"""
    system_prompt = [{"role": "system", "content": default_system_message}]
    
    # Ensure message formatting
    prompt_messages = system_prompt + [{"role": "user", "content": example["prompt"]}]
    chosen_messages = [{"role": "assistant", "content": example["chosen"]}]
    rejected_messages = [{"role": "assistant", "content": example["rejected"]}]

    # Apply chat template
    return {
        "prompt": tokenizer.apply_chat_template(prompt_messages, tokenize=False),
        "chosen": tokenizer.apply_chat_template(chosen_messages, tokenize=False),
        "rejected": tokenizer.apply_chat_template(rejected_messages, tokenize=False),
    }


# Load dataset from JSON
input_file = "/home/miria/cvxdpo/datasets/pref_dataset_alternate_edu_FINAL.json"
output_file = "test_edu_dataset.json"
print(len(output_file))

with open(input_file, "r", encoding="utf-8") as f:
    dataset = json.load(f)


# Convert to Hugging Face Dataset for easy processing
dataset = Dataset.from_list(dataset)

# select slive range here (13750) was og
dataset = dataset.shuffle().select(range(23750))
dataset = dataset.train_test_split(test_size=2750/13750)

# Apply transformation
dataset = dataset.map(apply_chat_template, fn_kwargs={"tokenizer": tokenizer})

# Save processed dataset
#dataset.to_json(output_file)

# save datasets to disk
dataset["train"].to_json("train_edu_dataset_subset.json", orient="records")
dataset["test"].to_json("test_edu_dataset_subset.json", orient="records")

print(f"Chat template applied and saved to {output_file}")


# import json
# import sys
# # Define the chat template format
# def apply_chat_template(example):
#     system_prompt = "<|im_start|>system\nYou are Dolphin, a helpful AI assistant.<|im_end|>\n"
#     #print(example)
#     #exit()
#     # Format prompt, chosen, and rejected responses
#     formatted_prompt = f"{system_prompt}<|im_start|>user\n{example['prompt']}<|im_end|>\n"
#     formatted_chosen = f"<|im_start|>assistant\n{example['chosen']}<|im_end|>\n"
#     formatted_rejected = f"<|im_start|>assistant\n{example['rejected']}<|im_end|>\n"

#     return {
#         "prompt": formatted_prompt,
#         "chosen": formatted_chosen,
#         "rejected": formatted_rejected
#     }

# # Load dataset
# input_file = "/home/miria/cvxdpo/datasets/pref_dataset_alternate_edu_FINAL.json"
# output_file = "ready.json"

# with open(input_file, "r", encoding="utf-8") as f:
#     dataset = json.load(f)

# # Apply chat template
# formatted_dataset = [apply_chat_template(example) for example in dataset]

# # Save the transformed dataset
# with open(output_file, "w", encoding="utf-8") as f:
#     json.dump(formatted_dataset, f, indent=4)

# print(f" Chat template applied and saved to {output_file}")