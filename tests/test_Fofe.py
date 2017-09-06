# Sample Test passing with nose and pytest

from wd_extractor.Corpus import Corpus
from wd_extractor.PreProcessor import PreProcessor
from wd_extractor.Fofe import Fofe
import pytest
import numpy as np
from pathlib import Path


@pytest.fixture(scope="module")
def corpus1():
    return Corpus('tests/data/fofe','fofetest.txt')


@pytest.fixture(scope="module")
def corpus2():
    return Corpus('tests/data/fofe','fofetest2.txt')

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
                           [0.0, 0.0, 0.0, 0.0, 1.0],
                           ])

@pytest.fixture(scope="module")
def rightFFMatrix():
    b = 0.9
    return np.array([[0.0, 0.0, b, b*b,   b*b*b],
                           [0.0, 0.0, 0.0, b,   b*b],
                           [0.0, 0.0, 0.0, 0.0, b],
                           [0.0, 0.0, 0.0, 0.0, 0.0],
                           [0.0, 0.0, 0.0, 0.0, 0.0],
                           ])

def test_data():
    path = Path('tests/data/fofe').glob('**/*.txt')
    assert path is not None

def testCurrentContext(corpus2, leftFFMatrix, focusFFMatrix, rightFFMatrix):
    docator = corpus2.getAllDocuments()
    doc = next(docator)
    a = 0.1
    b = 0.9
    fofe = Fofe(doc, a, b)

    left = fofe.leftContextFFMatrix()
    focus = fofe.focusContextFFMatrix(1)
    right = fofe.rightContextFFMatrix(1)

    assert np.allclose(left,leftFFMatrix)

    assert np.array_equal(focus,focusFFMatrix)

    assert np.array_equal(right, rightFFMatrix)

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
    fofe = Fofe(doc, a, b)
    data = fofe.doc2matrix()
    encoded = fofe.encode()

    assert np.array_equal(leftFFMatrix.dot(data), encoded[0])
    assert np.array_equal(focusFFMatrix.dot(data), encoded[1])
    assert np.array_equal(rightFFMatrix.dot(data), encoded[2])




'''
def test_fofe(corpus1):
    path = next(corpus1.getPathList())
    fofe = corpus1.makeFofe(0.7, path)
    assert fofe is not None
    s = fofe.__next__()
    assert np.allclose(s, [1.0, 0.0, 0.0])
    assert np.allclose(fofe.__next__(), [0.7, 1.0, 0.0])
    assert np.allclose(fofe.__next__(), [0.49, 0.7, 1.0])


def test_fofe2(corpus2):
    path = next(corpus2.getPathList())
    fofe = corpus2.makeFofe(0.7, path)
    assert fofe is not None
    for left_context in fofe:
        print(left_context)
'''