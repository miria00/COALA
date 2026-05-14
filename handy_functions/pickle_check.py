# import pickle

# def load_and_print_pickle(file_path):
#     """
#     Loads a pickle file and prints its contents.
    
#     Args:
#         file_path (str): Path to the pickle file.
#     """
#     try:
#         with open(file_path, 'rb') as f:
#             data = pickle.load(f)
#             print("Contents of the pickle file:")
#             print(data)
#     except Exception as e:
#         print(f"Error loading pickle file: {e}")

# # Replace 'your_file.pkl' with the path to your pickle file
# file_path = '...'
# load_and_print_pickle(file_path)


import json

# Load the JSON file
with open("/home/miria/cvxdpo/preference_tutor_dataset_small.json", "r") as file:
    data = json.load(file)

# Count the number of conversations
num_conversations = len(data)

# Print the result
print(f"Number of conversations in the JSON file: {num_conversations}")