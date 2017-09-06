import numpy as np
import tensorflow as tf

class Fofe:

    def __init__(self, document, leftFF, rightFF):
        self.document = document
        self.leftFF = leftFF
        self.rightFF = rightFF
        self.gramsize = 1
        self.leftFFMatrix = self.leftContextFFMatrix()
        self.focusFFMatrix = self.focusContextFFMatrix(1)
        self.rightFFMatrix = self.rightContextFFMatrix(1)
        self.docMatrix = self.doc2matrix()

    def focusContextFFMatrix(self, lookahead):
        length = self.document.length()
        i = np.identity(length)
        l = np.identity(length)

        for look in range(0, lookahead):
            l = self.shiftRight(l)
            i = i + l

        return i

    def leftContextFFMatrix(self):

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

    def rightContextFFMatrix(self, lookahead):
        l = self.leftPowersMatrix(self.document.length())
        r = l.T
        for look in range(0, lookahead):
            r = self.shiftRight(r)

        return self.powersToFofeMatrix(r, self.rightFF)

    def shiftRight(self,matrix):

        matrix = np.pad(matrix, ((0, 0), (0, 1)), 'constant')
        matrix = np.roll(matrix, 1)[:, :matrix.shape[1]-1]

        return matrix

    def doc2matrix(self):
        data = []
        for token in self.document.tokens:
            data.append(token.onehot())
        return np.array(data)

    def encode(self):

        doc = tf.Variable(self.docMatrix,name='doc')
        L = tf.Variable(self.leftFFMatrix, name='LeftFF')
        F = tf.Variable(self.focusFFMatrix, name='FocusFF')
        R = tf.Variable(self.rightFFMatrix, name='rightFF')
        init = tf.global_variables_initializer()

        left_f = tf.matmul(L,doc)
        focus_f = tf.matmul(F,doc)
        right_f = tf.matmul(R,doc)

        with tf.Session() as sess:
            init.run()
            left = left_f.eval()
            focus = focus_f.eval()
            right = right_f.eval()


        #left = self.leftFFMatrix.dot(self.docMatrix)
        #focus = self.focusFFMatrix.dot(self.docMatrix)
        #right = self.rightFFMatrix.dot(self.docMatrix)

        return left, focus, right





