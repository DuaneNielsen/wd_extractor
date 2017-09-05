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

def test_data():
    path = Path('tests/data/fofe').glob('**/*.txt')
    assert path is not None

def testCurrentContext(corpus2):
    docator = corpus2.getAllDocuments()
    doc = next(docator)
    fofe = Fofe()
    #print(fofe.currentContextMatrix(doc))
    left = fofe.leftContextMatrix(doc, 0.1)
    focus = fofe.focusContextMatrix(doc,1)
    right = fofe.rightContextMatrix(doc, 0.9, 1)

    print(right)

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