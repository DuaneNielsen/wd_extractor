import numpy as np


class Fofe:

    def __init__(self, document, leftFF, rightFF):
        self.document = document
        self.leftFF = leftFF
        self.rightFF = rightFF

    def focusContextMatrix(self, lookahead):
        length = self.document.length()
        i = np.identity(length)
        l = np.identity(length)

        for look in range(0, lookahead):
            l = self.shiftRight(l)
            i = i + l

        return i

    def leftContextMatrix(self):

        lm = self.leftPowersMatrix(self.document.length())
        i = self.powersToFofeMatrix(lm, self.leftFF)

        return i

    def leftPowersMatrix(self, length):
        l = []
        for row in range(0, length):
            r = []
            rv = row
            for column in range(0, length):
                r.append(rv)
                if rv > 0:
                    rv -= 1
            l.append(r)

        return np.matrix(l)

    def powersToFofeMatrix(self, powers_matrix, forget_factor):

        i = np.ones(powers_matrix.shape)
        i = i * forget_factor
        i = np.power(i, powers_matrix)

        for element in np.nditer(i, op_flags=['readwrite']):
            if element == 1.0:
                element[...] = 0.0

        return i

    def rightContextMatrix(self, lookahead):
        l = self.leftPowersMatrix(self.document.length())
        r = l.T
        for look in range(0, lookahead):
            r = self.shiftRight(r)

        return self.powersToFofeMatrix(r, self.rightFF)

    def shiftRight(self,matrix):

        matrix = np.pad(matrix, ((0, 0), (0, 1)), 'constant')
        matrix = np.roll(matrix, 1)[:, :matrix.shape[1]-1]

        return matrix

