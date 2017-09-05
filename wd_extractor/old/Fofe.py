from intervals import IntInterval
import numpy as np


class Fofe:

    def __init__(self,document, leftFF, rightFF):
        self.document = document
        self.focus_interval = IntInterval([0,0])
        self.leftFF = leftFF
        self.rightFF = rightFF
        self.left_context = None
        self.right_context = None
        self.encodeAllRight()

    def encodeToken(self, word_vector, context, forget_factor):
        if context is None:
            return word_vector
        else:
            new_context = (forget_factor * context)
            return np.add(new_context, word_vector)

    def unencodeToken(self,word_vector, context, forget_factor):
        new_context = np.subtract(context,word_vector)
        new_context = new_context / forget_factor
        return new_context

    def encodeRight(self):
        for token in reversed(self.document.tokens):
            word_vector = token.one_hot()
            self.right_context = self.encodeToken(word_vector, self.right_context, self.rightFF)

    def setInterval(self, interval):
        left = self.focus_interval.lower - interval.lower
        right = self.focus_interval.upper - interval.upper

        while left != 0:
            if left < 0:
                left += 1
                if self.document.hasNext(self.focus_interval.lower):
                    nextToken = self.document.nextToken(self.focus_interval.lower)
                    self.left_context = self.encodeToken(nextToken.one_hot(),)






