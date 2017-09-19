import scipy.sparse
import numpy

class FofeGramSet:
    def __init__(self, document, lookahead, leftContextMatrix, focusMatrices, rightContextMatrices):
        self.document = document
        self.lookahead = lookahead
        self.leftContextMatrix = leftContextMatrix
        self.focusMatrices = focusMatrices
        self.rightContextMatrices = rightContextMatrices
        self.labels = self.constructLabels()

    def addLabel(self, offset, lookahead, named_entity):
        """named_entity must be an Enum"""
        self.labels[offset,lookahead] = named_entity.value

    def constructLabels(self):
        num_rows = self.rightContextMatrices[0].shape[0]
        num_columns = len(self.rightContextMatrices)
        z = numpy.zeros((num_rows,num_columns),dtype=int)
        return scipy.sparse.lil_matrix(z)

    def saveLabels(self):
        scipy.sparse.save_npz(self.document.path + '.npz',self.labels)

    def getLabel(self, offset, lookahead):
        """returns an Enum with the label"""
        return self.document.corpus.label_types(self.labels[offset,lookahead])
