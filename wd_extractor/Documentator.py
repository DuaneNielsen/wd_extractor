from .Document import Document


class Documentator:

    def __init__(self, filepaths, corpus):
        self.filepaths = filepaths
        self.corpus = corpus
        self.path = None

    def __iter__(self):
        return self

    def __next__(self):
        self.path = next(self.filepaths,None)
        if self.path is None:
            raise StopIteration
        return Document(self.corpus,self.path,None)
