import numpy as np

class Vocabulary:

    def __init__(self):
        self.vocab = {}

    def add(self, token):
        self.vocab[token.word] = 1.0

    def buildOneHotLookup(self):
        one_hot_length = len(self.vocab)
        diag_square_matrix = np.identity(one_hot_length,float)

        i = 0
        for word in self.vocab:
            self.vocab[word] = diag_square_matrix[i]
            i = i+1

    def onehot(self, token):
        return self.vocab[token.word]

    def length(self):
        return self.vocab.__len__()