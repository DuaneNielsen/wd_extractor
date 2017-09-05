
from pycorenlp import StanfordCoreNLP
from .Token import Token


class Tokenizer():

    def __init__(self):
        self.nlp = StanfordCoreNLP('http://localhost:9000')

    # if the token is $100.0 or some number
    # then replace it with NUMBER
    def tokenGetWord(self, token):
        if token['ner'] in set(['MONEY', 'NUMBER']):
            return token['ner']
        else:
            return token['word']

    def tokens(self, document):
        tokens = []
        text = document.getText()
        output_json = self.nlp.annotate(text, properties={
            'annotators': 'tokenize,ssplit,ner',
            'outputFormat': 'json'
        })
        for sentence in output_json['sentences']:
            for token in sentence['tokens']:
                word = self.tokenGetWord(token)
                text = token['word']
                tokens.append(Token(word,document,text,token))

        return tokens