'''
This file creates the DPO training dataset format
reads data from a .json file, processes the conversations between two agents (agent1 and agent2), and generates dataset dictionaries for DPO
extract utterances from the conversation and splits them into agent1 and agent2 responses
creates a dictionary where prompt holds the prompts (agent2's statements), and 'chosen' and 'rejected' represent ALTERNATE responses from agent1
return final list of these dictionaries
optional: print the generated dataset dictionaries.
agent1 = tutor
agent2 = student
'''


import json
import pprint
import pandas as pd
# import os
# import jax
# import jax.numpy as jnp
import json

DATASET_PATH = 'preference_tutor_dataset_merged.json'
OUTPUT_PATH = "pref_datset_alternate_FINAL.json"  # Output file


def get_dataset_dicts():
    """
    Processes dataset and prepares it in a format compatible with JAX.
    creates a list of dictionaries with: prompt, chosen, rejected
    """
    list_of_dataset_dicts = []

    with open(DATASET_PATH, 'r') as f:
        json_data = json.load(f)
        for conversation in json_data:
            dpo_dataset_dict = {'prompt': [], 'chosen': [], 'rejected': []}
            agent1 = []
            agent2 = []
            for utterance in conversation['utterances']:
                if utterance[0] == 'agent1':
                    agent1.append(utterance[1])
                else:
                    agent2.append(utterance[1])

            # Sometimes agent2 has the final say... ignore the final prompt
            if len(agent1) == len(agent2):
                agent2 = agent2[:-1]

            # each agent2 msg is a prompt
            for i, prompt in enumerate(agent2):
                # The full prompt includes the entire convo till this latest prompt
                full_prompt = dpo_dataset_dict['prompt'][-1] if len(dpo_dataset_dict['prompt']) > 0 else ''
                if i == 0:
                    full_prompt += f'agent2: {prompt}'
                else:
                    full_prompt += f'\nagent1: {agent1[i]}\nagent2: {prompt}'


                dpo_dataset_dict['prompt'].append(full_prompt)
                dpo_dataset_dict['chosen'].append(agent1[i+1])
                dpo_dataset_dict['rejected'].append(agent1[(i+2) % len(agent1)])

            for p,c,r in zip(dpo_dataset_dict['prompt'], dpo_dataset_dict['chosen'], dpo_dataset_dict['rejected']):
                single_dict = {'prompt': p, 'chosen': c, 'rejected': r}
                list_of_dataset_dicts.append(single_dict)

    return list_of_dataset_dicts



if __name__ == '__main__':
    dataset_dicts = get_dataset_dicts()

    # Save dataset to JSON file
    with open(OUTPUT_PATH, 'w') as out_f:
        json.dump(dataset_dicts, out_f, indent=4)
    # print(dataset_dicts)

    # pprint.pp(dataset_dicts[0])

    # print("dataframe check starts here----------")
    # df = pd.DataFrame(dataset_dicts)

    # # Adding the lengths of the prompts, chosen, and rejected sets
    # df['prompt_length'] = df['prompt'].apply(len)
    # df['chosen_length'] = df['chosen'].apply(len)
    # df['rejected_length'] = df['rejected'].apply(len)

    # print(df[['prompt', 'chosen', 'rejected', 'prompt_length', 'chosen_length', 'rejected_length']])