from wd_extractor.Corpus import Corpus
from pathlib import PosixPath,Path
import pytest

@pytest.fixture(scope="module")
def corpus():
    pathlist = Path('tests/data/corpus/').glob('*.txt')
    path = next(pathlist)
    assert path == PosixPath('tests/data/corpus/vocabtest.txt')
    return Corpus('tests/data/corpus/', '*.txt')


def setup_module():
    print('setup')


def testSimpleCorpus(corpus):
    assert corpus.vocab.length() == 18
    return None


def testDocumentText(corpus):
    doc = None
    for document in corpus.getAllDocuments():
        doc = document
    assert doc is not None
    assert doc.getText() == 'this is the basic vocab\nthis is the next vocab\nyou owe me $20\nthis is a floating point number with a probability of 1.0\n$20 1.0 0.2 10'
    tokens = doc.corpus.tokenizer.tokens(doc)
    assert tokens[0].word == 'this'
    assert tokens[1].word == 'is'
    assert tokens[13].word == 'MONEY'


def testGrams(corpus):
    documentator = corpus.getAllDocuments()
    doc = next(documentator)
    assert doc is not None
    for ngram in doc.nGrams(5):
        for gram in ngram:
            print(gram.interval)