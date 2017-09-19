# Sample Test passing with nose and pytest

from wd_extractor.Corpus import Corpus
from wd_extractor.PreProcessor import PreProcessor
from wd_extractor.Fofe import Fofe
import pytest
import numpy as np
from pathlib import Path
import json
from enum import Enum


class Labels(Enum):
    COMPANY = 1
    PERSON = 2


@pytest.fixture(scope="module")
def corpus1():
    return Corpus('tests/data/fofe','fofetest.txt',Labels)


@pytest.fixture(scope="module")
def corpus2():
    return Corpus('tests/data/fofe','fofetest2.txt',Labels)


@pytest.fixture(scope="module")
def leftFFMatrix():
    a = 0.1
    return np.array([[0.0,0.0,0.0,0.0,0.0],
                          [a,0.0,0.0,0.0,0.0],
                          [a*a,a,0.0,0.0,0.0],
                          [a*a*a,a*a,a,0.0,0.0],
                          [a*a*a*a,a*a*a,a*a,a,0.0]])


@pytest.fixture(scope="module")
def focusFFMatrix():
    return np.array([[1.0, 1.0, 0.0, 0.0, 0.0],
                      [0.0, 1.0, 1.0, 0.0, 0.0],
                      [0.0, 0.0, 1.0, 1.0, 0.0],
                      [0.0, 0.0, 0.0, 1.0, 1.0],
                      [0.0, 0.0, 0.0, 0.0, 1.0]])


@pytest.fixture(scope="module")
def rightFFMatrix():
    b = 0.9
    return np.array([[0.0, 0.0, b, b*b,   b*b*b],
                           [0.0, 0.0, 0.0, b,   b*b],
                           [0.0, 0.0, 0.0, 0.0, b],
                           [0.0, 0.0, 0.0, 0.0, 0.0],
                           [0.0, 0.0, 0.0, 0.0, 0.0],
                           ])


@pytest.fixture(scope="module")
def shifted2():
    return np.array([[0,0,1,0,0],
                     [0,0,0,1,0],
                     [0,0,0,0,1],
                     [0,0,0,0,0],
                     [0,0,0,0,0]])


@pytest.fixture(scope="module")
def doc(corpus1):
    docator = corpus1.getAllDocuments()
    return next(docator)

@pytest.fixture(scope="module")
def doc2(corpus2):
    docator = corpus2.getAllDocuments()
    return next(docator)

@pytest.fixture(scope="module")
def fofe(doc):
    return Fofe(doc,0.1,0.9,0)

@pytest.fixture(scope="module")
def fofe2(doc2):
    return Fofe(doc2,0.1,0.9,1)

@pytest.fixture(scope="module")
def labels():
    handle = open("tests/data/fofe/fofetest2.label", "r")
    text = handle.read()
    return json.loads(text)

def test_data():
    path = Path('tests/data/fofe').glob('**/*.txt')
    assert path is not None

def testCurrentContext(corpus2, leftFFMatrix, focusFFMatrix, rightFFMatrix):
    docator = corpus2.getAllDocuments()
    doc = next(docator)
    a = 0.1
    b = 0.9
    fofe = Fofe(doc, a, b, 1)

    left = fofe.leftContextFFMatrix()
    focus = fofe.focusContextFFMatrix(1)
    right = fofe.rightContextFFMatrix(1)

    assert np.allclose(left,leftFFMatrix)
    assert np.array_equal(focus[1],focusFFMatrix)
    assert np.array_equal(right[1], rightFFMatrix)


def testDoc2Matrix(corpus1):
    docator = corpus1.getAllDocuments()
    doc = next(docator)
    a = 0.1
    b = 0.9
    fofe = Fofe(doc, a, b)
    data = fofe.doc2matrix()
    identity = np.identity(3)
    assert np.array_equal(data, identity)


def testEncode(corpus2, leftFFMatrix, focusFFMatrix, rightFFMatrix):
    docator = corpus2.getAllDocuments()
    doc = next(docator)
    a = 0.1
    b = 0.9
    fofe = Fofe(doc, a, b, 1)
    data = fofe.doc2matrix()
    encoded = fofe.encode()

    assert np.array_equal(leftFFMatrix.dot(data), encoded.leftContextMatrix)
    assert np.array_equal(focusFFMatrix.dot(data), encoded.focusMatrices[1])
    assert np.array_equal(rightFFMatrix.dot(data), encoded.rightContextMatrices[1])


def testSubFuncs(fofe):
    left_powers = fofe.leftPowersMatrix(3)
    fofe_values = fofe.powersToFofeMatrix(left_powers, 0.5)
    assert left_powers is not None
    assert fofe_values is not None



def testShift(fofe,shifted2):
    eye = np.identity(5)
    eye = fofe.shiftRight(eye,2)
    assert(np.array_equal(eye, shifted2))


def testLabels(fofe2):
    gramset = fofe2.encode()
    assert gramset.labels.shape == (5,2)
    gramset.addLabel(0,0,Labels.COMPANY)
    gramset.addLabel(0,1,Labels.PERSON)
    gramset.addLabel(1,0,Labels.COMPANY)
    gramset.addLabel(1,1,Labels.PERSON)
    assert gramset.getLabel(0,0) == Labels.COMPANY
    assert gramset.getLabel(0,1) == Labels.PERSON
    assert gramset.getLabel(1,0) == Labels.COMPANY
    assert gramset.getLabel(1,1) == Labels.PERSON


