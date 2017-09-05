import numpy as np


class Fofe:

    #def __init__(self):





    def focusContextMatrix(self, document, lookahead):
        length = document.length()
        i = np.identity(length)
        l = np.identity(length)

        #lookahead += 1

        #for look in range(1, lookahead):
        #    l = np.identity(length - look)
        #    l = np.pad(l, look, 'constant')
        #    l = l[look:, :5]
        #    i = i + l

        for look in range(0, lookahead):
            l = self.shiftRight(l)
            i = i + l

        return i

    def leftContextMatrix(self, document, forget_factor):

        lm = self.leftPowersMatrix(document.length(), forget_factor)
        i = self.powersToFofeMatrix(lm, forget_factor)

        return i

    def leftPowersMatrix(self, length, forget_factor):
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

    def rightContextMatrix(self, document, forget_factor, lookahead):
        l = self.leftPowersMatrix(document.length(), forget_factor)
        r = l.T
        for look in range(0, lookahead):
            r = self.shiftRight(r)

        return self.powersToFofeMatrix(r, forget_factor)

    def shiftRight(self,matrix):
        matrix = np.pad(matrix, ((0, 0), (0, 1)), 'constant')
        matrix = np.roll(matrix, 1)[:, :matrix.shape[1]-1]
        return matrix

