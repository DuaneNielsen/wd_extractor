from .Gram import Gram
from intervals import IntInterval


class NGram:
    def __init__(self, document, interval):
        self.document = document
        self.interval = interval

    def __iter__(self):
        return self

    def __next__(self):
        if self.interval.lower < self.interval.upper:
            gram = Gram(self.document, IntInterval([self.interval.lower, self.interval.upper]), None)
            self.interval.upper -= 1
            return gram
        else:
            raise StopIteration
