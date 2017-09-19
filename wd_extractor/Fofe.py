import numpy as np
import tensorflow as tf
from .FofeGramSet import FofeGramSet

class Fofe:

    def __init__(self, document, leftFF, rightFF, lookahead=0):
        self.document = document
        self.leftFF = leftFF
        self.rightFF = rightFF
        self.lookahead = lookahead
        self.leftFFMatrix = self.leftContextFFMatrix()
        self.focusFFMatrices = self.focusContextFFMatrix(lookahead)
        self.rightFFMatrices = self.rightContextFFMatrix(lookahead)
        self.docMatrix = self.doc2matrix()

    def focusContextFFMatrix(self, lookahead):
        focusMatrices = []
        length = self.document.length()
        i = np.identity(length)
        l = np.identity(length)

        focusMatrices.append(i)

        for look in range(0, lookahead):
            l = self.shiftRight(l)
            i = i + l
            focusMatrices.append(i)

        return focusMatrices

    def leftContextFFMatrix(self):
        lm = self.leftPowersMatrix(self.document.length())
        i = self.powersToFofeMatrix(lm, self.leftFF)

        return i



    def leftPowersMatrix(self, length):
        """Returns length x length matrix in the form
            [[0 0 0]
             [1 0 0]
             [2 1 0]]"""

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
        """Returns accepts matrix of exponents, and returns a matrix with elements = forgetfactor ^ exponent
            [[0 0 0]                [ 0.    0.    0.  ]
             [1 0 0]    *  0.5   =  [ 0.5   0.    0.  ]
             [2 1 0]]               [ 0.25  0.5   0.  ]
        """

        i = np.ones(powers_matrix.shape)
        i = i * forget_factor
        i = np.power(i, powers_matrix)

        for element in np.nditer(i, op_flags=['readwrite']):
            if element == 1.0:
                element[...] = 0.0

        return i

    def rightContextFFMatrix(self, lookahead):
        rMatrices = []
        l = self.leftPowersMatrix(self.document.length())
        r = l.T

        for i in range(0, lookahead+1):
            shifted = self.shiftRight(r,i)
            rMatrices.append(self.powersToFofeMatrix(shifted,self.rightFF))

        return rMatrices

    def shiftRight(self,matrix, columns=1):

        matrix = np.pad(matrix, ((0, 0), (0, columns)), 'constant')
        matrix = np.roll(matrix, columns)[:, :matrix.shape[1]-columns]

        return matrix

    def doc2matrix(self):
        data = []
        for token in self.document.tokens:
            data.append(token.onehot())
        return np.array(data)

    def doc2labels(self):
        data = []
        for ngram in self.document.nGrams(self.lookahead):
            for gram in ngram:
                data.append()

    def encode(self):

        doc = tf.Variable(self.docMatrix,name='doc')
        L = tf.Variable(self.leftFFMatrix, name='LeftFF')

        F = []
        R = []

        for i in range(0,self.lookahead+1):
            F.append(tf.Variable(self.focusFFMatrices[i], name='FocusFF_look_' + str(i)))
            R.append(tf.Variable(self.rightFFMatrices[i], name='RightFF_look_' + str(i)))

        init = tf.global_variables_initializer()

        left_f = tf.matmul(L,doc)
        focus_f = []
        right_f = []

        for i in range(0,self.lookahead+1):
            focus_f.append(tf.matmul(F[i],doc))
            right_f.append(tf.matmul(R[i],doc))

        with tf.Session() as sess:
            init.run()
            left = left_f.eval()

            focus = []
            right = []
            for i in range(0,self.lookahead+1):
                focus.append(focus_f[i].eval())
                right.append(right_f[i].eval())

            return FofeGramSet(self.document, self.lookahead, left, focus, right)





