from wd_extractor.Corpus import Corpus
from wd_extractor.PreProcessor import PreProcessor
from pathlib import PosixPath
import pytest

@pytest.fixture(scope="module")
def corpus():
    p = PreProcessor()
    return Corpus(p, 'tests/data/corpus/', '*.txt')


def setup_module():
    print('setup')


def testSimpleCorpus(corpus):
    assert corpus.vocab.__len__() == 18


def testFofe(corpus):
    path = next(corpus.getPathList())
    assert path == PosixPath('tests/data/corpus/vocabtest.txt')
    assert corpus.makeFofe(0.7, path) is not None
