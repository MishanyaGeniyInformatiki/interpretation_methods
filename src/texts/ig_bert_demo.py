import torch
import numpy as np
import json
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

from omnixai.data.text import Text
from omnixai.explainers.nlp.specific.ig import IntegratedGradientText

from texts.utils.models import BertWrapper
from texts.utils.utils import BertPreprocess, save_explanation, load_explanation


if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
    model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

    model = BertWrapper(model)

    # my_text = ["best film ever", "it was a horrible movie"]
    # my_text_t = Text(my_text, tokenizer=tokenizer)

    # Load example
    preprocess = BertPreprocess(tokenizer)
    # print(preprocess(my_text_t))

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
        torch_inputs.append(inp)

    with torch.no_grad():
        output = model(*torch_inputs).softmax(dim=1)
    print(output.tolist()[0])

    # Explaining
    explainer = IntegratedGradientText(
        model=model,
        embedding_layer=model.bert.base_model.embeddings,
        preprocess_function=preprocess,
        id2token=tokenizer.ids_to_tokens
    )

    class_names = ["negative", "positive"]
    explanation = explainer.explain(x_example_text)

    explanation_name = "explanation_bert"
    explanation_path = "../texts/explanation_results/" + explanation_name
    save_explanation(explanation, explanation_path)

    expl = load_explanation(explanation_path)

    fig = expl.plotly_plot(class_names=class_names)
    fig.show()
