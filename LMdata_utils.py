"""
This file contains handy custom imdb datasets utils
class MovieReviewsDataset and class Gpt2ClassificationCollator

this is a subset of LM_complete_utils.py

todo: add support for mistral and llama into same file, change name of functions to generic
"""

import io
import os
import torch
from torch.utils.data import Dataset

'''
pytorch dataset needs 3 components: 
- init() where we read in data and transform text into numbers
- len() where we return the number of examples
- getitem() given an int, returns the example indexed at that position
'''

'''
3 parts of GPT2 we need to use:
- config (GPTConfig)
- tokenizer (GPT2Tokenizer)
- model (GPT2ForSequenceClassification)
'''

class Gpt2ClassificationCollator(object):
    """
    Data Collator (parser) used for GPT2 in a classification task.
    """

    def __init__(self, use_tokenizer, labels_encoder, max_sequence_len=None):
        self.use_tokenizer = use_tokenizer
        self.max_sequence_len = use_tokenizer.model_max_length if max_sequence_len is None else max_sequence_len
        self.labels_encoder = labels_encoder

    def __call__(self, sequences):
        texts = [sequence['text'] for sequence in sequences]
        labels = [sequence['label'] for sequence in sequences]
        inputs = self.use_tokenizer(text=texts, return_tensors="pt", padding=True, truncation=True, max_length=self.max_sequence_len)
        inputs.update({'labels': torch.tensor(labels)})
        return inputs


# -----------------------------------------------------------------------------
# utils for imdb data follow

class Gpt2ClassificationCollator_simple(object):

    def __init__(self, use_tokenizer, labels_encoder, max_sequence_len=None):

        # tokenizer to be used inside the class.
        self.use_tokenizer = use_tokenizer
        # check max sequence length.
        self.max_sequence_len = use_tokenizer.model_max_length if max_sequence_len is None else max_sequence_len
        # label encoder used inside the class.
        self.labels_encoder = labels_encoder

        return

    def __call__(self, sequences):
        texts = [sequence['text'] for sequence in sequences]
        inputs = self.use_tokenizer(text=texts, return_tensors="pt", padding=True, truncation=True,  max_length=self.max_sequence_len)

        return inputs
    

class MovieReviewsDataset_simple(Dataset): 

  def __init__(self, path, use_tokenizer):

    if not os.path.isdir(path):
      
      raise ValueError('Invalid path! Needs to be a directory')
    self.texts = []

    sentiment_path = path

    files_names = os.listdir(sentiment_path)
    for file_name in files_names:
      file_path = os.path.join(sentiment_path, file_name)

      # parse content
      content = io.open(file_path, mode='r', encoding='utf-8').read()
      self.texts.append(content)

    # num of exmaples
    self.n_examples = len(self.texts)
    return

  def __len__(self):
    return self.n_examples

  def __getitem__(self, item):
    return {'text':self.texts[item]}