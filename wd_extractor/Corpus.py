from pathlib import Path
from pycorenlp import StanfordCoreNLP
from .Fofe import Fofe


class Corpus:

    def __init__(self, tokenizer, directory, fileregex ):
        self.tokenizer = tokenizer
        self.nlp = StanfordCoreNLP('http://localhost:9000')
        self.directory = directory
        self.vocab = {}
        pathlist = Path(directory).glob(fileregex)
        for path in pathlist:
            print('**********')
            print(path)
            print('**********')
            # because path is object not string
            handle = open(path, "r")
            text = handle.read()
            output_json = self.nlp.annotate(text, properties={
                'annotators': 'tokenize,ssplit,ner',
                'outputFormat': 'json'
            })

            # init vocabulary
            for sentence in output_json['sentences']:
                for token in sentence['tokens']:
                    self.vocab[self.tokenizer.tokenGetWord(token)] = 1

    def makeFofe(self, forgetFactor, path):
        return Fofe(self.vocab, self.tokenizer, forgetFactor, path)

    def getPathList(self):
        return Path(self.directory).glob('**/*.txt')

