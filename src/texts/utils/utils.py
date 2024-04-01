import json
import numpy as np

from omnixai.data.text import Text
from omnixai.explanations.text.word_importance import WordImportance

max_length = 256


class Preprocess:
    def __init__(self, transformer):
        self._transformer = transformer

    def __call__(self, x_inp: Text, **kwargs):
        samples = self._transformer.transform(x_inp)
        max_len = 0
        for i in range(len(samples)):
            max_len = max(max_len, len(samples[i]))
        max_len = min(max_len, max_length)
        inputs = np.zeros((len(samples), max_len), dtype=int)
        masks = np.zeros((len(samples), max_len), dtype=np.float32)
        for i in range(len(samples)):
            x = samples[i][:max_len]
            inputs[i, :len(x)] = x
            masks[i, :len(x)] = 1
        return inputs, masks


class BertPreprocess:
    def __init__(self, tokenizer):
        self._tokenizer = tokenizer

    def __call__(self, x_inp: Text, **kwargs):
        tokenizer_output = self._tokenizer(x_inp.values)
        # tokenizer_output['attention_mask'] # create 'attention_mask in this function (const shape)
        samples = tokenizer_output['input_ids']

        max_len = 0
        for i in range(len(samples)):
            max_len = max(max_len, len(samples[i]))
        max_len = min(max_len, max_length)
        inputs = np.zeros((len(samples), max_len), dtype=int)
        masks = np.zeros((len(samples), max_len), dtype=np.float32)
        for i in range(len(samples)):
            x = samples[i][:max_len]
            inputs[i, :len(x)] = x
            masks[i, :len(x)] = 1
        return inputs, masks


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


def save_explanation(explanation, path):
    with open(path, "w") as file:
        json.dump({'explanation': explanation.explanations}, file, indent=4, cls=NpEncoder)
    file.close()


def load_explanation(path):
    with open(path, 'r') as file:
        data_explanations = json.load(file)
    file.close()

    explanation = WordImportance(mode="classification", explanations=data_explanations['explanation'])
    return explanation
