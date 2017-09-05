from intervals import IntInterval
from .NGram import NGram


class Graminator:

    def __init__(self, document, gramsize):
        self.document = document
        self.gramsize = gramsize
        self.interval = IntInterval([0,gramsize])
        self.lower = self.interval.lower
        self.upper = self.interval.upper

    def __iter__(self):
        return self

    def __next__(self):
        if self.lower < self.document.length():
            self.upper = self.lower + self.maxLookahead(self.lower, self.gramsize)
            ngram_int = IntInterval([self.lower, self.upper])
            ngram = NGram(self.document, ngram_int)
            self.lower += 1
            return ngram
        else:
            raise StopIteration

    def maxLookahead(self, cursor, lookahead):
        if cursor + lookahead > (self.document.length() - 1):
            return self.document.length() - cursor
        else:
            return lookahead