from .Graminator import Graminator


class Document:

    def __init__(self, corpus, path, grams):

        self.corpus = corpus
        self.grams = grams
        self.graminator = None
        self.path = path
        self.tokens = corpus.tokenizer.tokens(self)

    def getText(self):
        if self.path is not None:
            handle = open(self.path, "r")
            text = handle.read()
            return text

    def length(self):
        return len(self.tokens)

    def nGrams(self, gramsize):
        return Graminator(self, gramsize)

    def hasNext(self, index):
        index += 1
        return (index > 0) and index < len(self.tokens)

    def nextToken(self, index):
        return self.tokens[index + 1]

    def hasPrev(self, index):
        index -= 1
        return (index > 0) and index < len(self.tokens)

    def prevToken(self, index):
        return self.tokens[index-1]
