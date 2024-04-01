from nltk.tokenize import TweetTokenizer

"""
Здесь нужно определить токенизатор. Возможно использовать любой токенизатор, 
который будет принимать строку или массив строк на вход и возвращать массив 
токенов для каждой строки. (см. метод to_tokens класса Text). 
Если не указывать токенизатор при создании объекта класса Text, 
то будет использоваться стандартный nltk токенизатор. (см. метод to_tokens класса Text)
"""


class TokenizerMy:
    def __init__(self):
        self._tok = TweetTokenizer()

    def __call__(self, *args, **kwargs):
        # return [self._tok.tokenize(s.lower()) for s in args[0]]
        return [self._tok.tokenize(s) for s in args[0]]
