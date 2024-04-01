import torch.nn as nn

"""
Требования: наличие параметра embedding в архитектуре модели 
(The embedding layer in the model, which can be torch.nn.Module). 
В качестве примера зачастую используется класс nn.Embedding в PyTorch 
для преобразования индексов слов в плотные векторы фиксированного размера, 
называемые встраиваниями.
"""


class ModelMy(nn.Module):
    def __init__(self, vocab_size, embedding_size, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.fc = nn.Linear(embedding_size, num_classes)

    def forward(self, input_ids, mask):
        x = self.embedding(input_ids)
        x = x * mask.unsqueeze(dim=-1)
        x = x.mean(dim=1)
        return self.fc(x)


class BertWrapper(nn.Module):
    def __init__(self, bert: nn.Module):
        super().__init__()
        self.bert = bert

    def forward(self, input_ids, mask):
        input_ = {'input_ids': input_ids, 'attention_mask': mask}
        out = self.bert(**input_).logits
        return out
