from pathlib import Path
from pycorenlp import StanfordCoreNLP
from .Vocabulary import Vocabulary
from .Documentator import Documentator
from .Tokenizer import Tokenizer

class Corpus:

    def __init__(self, directory, fileregex, label_types):
        """label_types is an enum"""
        self.tokenizer = Tokenizer()
        self.directory = directory
        self.vocab = Vocabulary()
        self.directory = directory
        self.fileregex = fileregex
        self.label_types = label_types

        for document in self.getAllDocuments():
            for token in document.tokens:
                self.vocab.add(token)
        self.vocab.buildOneHotLookup()

    def getAllDocuments(self):
        filepaths = Path(self.directory).glob(self.fileregex)
        return Documentator(filepaths, self)

    def getLabeledDocuments(self):
        return None
