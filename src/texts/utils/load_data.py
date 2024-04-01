import pandas as pd
import numpy as np

from omnixai.data.text import Text


def load_data(tokenizer):

    class_names = ["negative", "positive"]

    data_path = '../texts/data/sst2/splited'

    x_train = Text(np.squeeze(pd.read_csv(data_path + "/x_train.csv").values[:]).tolist(), tokenizer=tokenizer)
    y_train = np.squeeze(pd.read_csv(data_path + "/y_train.csv").values[:].astype(int))
    x_test = Text(np.squeeze(pd.read_csv(data_path + "/x_test.csv").values[:]).tolist(), tokenizer=tokenizer)
    y_test = np.squeeze(pd.read_csv(data_path + "/y_test.csv").values[:].astype(int))

    return x_train, y_train, x_test, y_test, class_names
