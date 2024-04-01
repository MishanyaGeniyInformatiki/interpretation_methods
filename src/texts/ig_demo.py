import torch
import torch.nn as nn
import sklearn
import numpy as np

from omnixai.data.text import Text
from omnixai.preprocessing.text import Word2Id
from omnixai.explainers.tabular.agnostic.L2X.utils import Trainer, InputData, DataLoader
from omnixai.explainers.nlp.specific.ig import IntegratedGradientText

from texts.utils.tokenizer import TokenizerMy
from texts.utils.load_data import load_data
from texts.utils.models import ModelMy
from texts.utils.utils import Preprocess, save_explanation, load_explanation

"""
Text
Для работы со строковыми данными (текстом, предложениями) используется класс Text. 
Для инициализации объекта класса необходимо передать в конструктор класса Text саму 
строку текста или массив строк, а вторым параметром - токенизатор, который разбивает строку на токены.
"""

"""
Transformer
Здесь нужно определить свой трансформер.
Переводит токены в числа, хранит большой словарь {токен: число}.
Класс трансформера должен иметь атрибуты word_to_id = {слово: id слова в словаре} и id_to_word = {id: слово}
Пример трансформера - класс Word2Id. Имеет метод fit - принимает на вход объект класса Text (обычно x_train) и 
фиксирует на нем словарь (создает словари word_to_id и id_to_word).
Имеет метод transform, который переводит массив строковых данных в числа, используя словарь токенов и айдишников.
"""

max_length = 256


if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = TokenizerMy()

    x_train, y_train, x_test, y_test, class_names = load_data(tokenizer)

    # используем предложенный библиотекой omnixai трансформер Word2Id
    transformer = Word2Id()
    print("Fitting transformer ... ")
    transformer.fit(x_train)
    print("Done!\n")

    # Define model
    model = ModelMy(
        vocab_size=transformer.vocab_size,
        embedding_size=50,
        num_classes=len(class_names)
    ).to(device)

    # Train model
    print("Training model ... ")
    Trainer(
        optimizer_class=torch.optim.AdamW,
        learning_rate=1e-3,
        batch_size=128,
        num_epochs=50,
    ).train(
        model=model,
        loss_func=nn.CrossEntropyLoss(),
        train_x=transformer.transform(x_train),
        train_y=y_train,
        padding=True,
        max_length=max_length,
        verbose=True
    )
    print("Model trained!\n")

    # Validation on test
    model.eval()
    data = transformer.transform(x_test)
    data_loader = DataLoader(
        dataset=InputData(data, [0] * len(data), max_length),
        batch_size=32,
        collate_fn=InputData.collate_func,
        shuffle=False
    )
    outputs = []
    for inp in data_loader:
        value, mask, target = inp
        y = model(value.to(device), mask.to(device))
        outputs.append(y.detach().cpu().numpy())
    outputs = np.concatenate(outputs, axis=0)
    predictions = np.argmax(outputs, axis=1)
    print('Test accuracy: {}'.format(sklearn.metrics.f1_score(y_test, predictions, average='binary')))
    # ------------------ #

    # Load example
    preprocess = Preprocess(transformer)

    with open('../texts/data/test_examples/review.txt', 'r', encoding='utf-8') as file:
        data = file.readlines()
    file.close()

    x_example_list = data
    print(x_example_list)

    x_example_text = Text(data, tokenizer=tokenizer)
    inputs = preprocess(x_example_text)

    torch_inputs = []
    for inp in inputs:
        if isinstance(inp, (np.ndarray, list)):
            inp = torch.tensor(inp)
        torch_inputs.append(inp.to(device))

    with torch.no_grad():
        output = model(*torch_inputs).softmax(dim=1)
    print(output.tolist()[0])

    # Explaining
    explainer = IntegratedGradientText(
        model=model,
        embedding_layer=model.embedding,
        preprocess_function=preprocess,
        id2token=transformer.id_to_word
    )

    explanation = explainer.explain(x_example_text)

    explanation_name = "explanation"
    explanation_path = "../texts/explanation_results/" + explanation_name
    save_explanation(explanation, explanation_path)

    expl = load_explanation(explanation_path)

    fig = expl.plotly_plot(class_names=class_names)
    fig.show()
