import numpy as np
from pycorenlp import StanfordCoreNLP


class Fofe:

    def __init__(self, vocab, pre_processor, forget_factor, path):
        self.left_context = None
        self.vocab = vocab
        self.pre_processor = pre_processor
        self.forget_factor = forget_factor
        self.one_hot_length = len(vocab)
        # encode each word into one-hot using diagonal square matrix
        self.word_vectors = np.identity(self.one_hot_length, float)
        self.nlp = StanfordCoreNLP('http://localhost:9000')

        # set the values of the vocab to the one-hot encoding
        i = 0
        for word in vocab:
            vocab[word] = self.word_vectors[i]
            i = i+1

        # load up the document to be encoded
        handle = open(path, "r")
        text = handle.read()
        output_json = self.nlp.annotate(text, properties={
            'annotators': 'tokenize,ssplit,ner',
            'outputFormat': 'json'
        })

        self.tokens = []
        self.token = None

        for sentence in output_json['sentences']:
            for token in sentence['tokens']:
                self.tokens.append(token)

        self.token = iter(self.tokens)

    def __iter__(self):
        return self

    def getOneHotLength(self):
        return self.one_hot_length

    def lookupOneHot(self, word):
        return self.vocab[self.pre_processor.tokenGetWord(word)]

    def __next__(self):
        word_vector = self.lookupOneHot(next(self.token))
        if self.left_context is None:
            self.left_context = word_vector
            return word_vector
        else:
            self.left_context = (self.forget_factor * self.left_context)
            self.left_context = np.add(self.left_context, word_vector)
            return self.left_context

    def fofe_calc(self, context, word_vector):
        context = context * self.forget_factor
        return np.add(context, word_vector)

    #needs fixing
    def fofe_lookahead(self, words, lookahead):
        lookahead_state = []
        i = 0
        for word in words[1:lookahead]:
            lookahead_state[i] = self.fofe_calc(lookahead_state, word)
            i=i+1
        return self.left_context, lookahead_state


    def fofe_decode(_self):
        return state
    # actually the decoder is not trivial unless forget factor is less than or equal 0.5
    # and assume the string does not contain repeated words
    # if this holds (unlikely) then decoding will amount to
    # find the one-hot with value = 1.0
    # lookup the value in the vocab
    # subtract 1.0 from the value and multiply by 1/forget factor


