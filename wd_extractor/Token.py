class Token:

    def __init__(self, word, document, text, attributes):
        self.text = text
        self.document = document
        self.word = word
        self.attributes = attributes

    def one_hot(self):
        return self.document.corpus.vocab.one_hot(self)