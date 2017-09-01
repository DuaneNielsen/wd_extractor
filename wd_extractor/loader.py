from pathlib import Path
from pycorenlp import StanfordCoreNLP
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
import keras
import logging
import sys
from PreProcessor import PreProcessor
from Fofe import Fofe

log = logging.getLogger("fofeparser")

sh = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
sh.setFormatter(formatter)
log.addHandler(sh)
log.setLevel(logging.DEBUG)

#active_named_entities = set(['MONEY', 'NUMBER'])


# if the token is $100.0 or some number
# then replace it with NUMBER
# def tokenGetWord(token):
#     if token['ner'] in active_named_entities:
#         return token['ner']
#     else:
#         return token['word']


def fofe(state, s):
    if state is None:
        return s
    else:
        stuff = (forget_factor * state)
        return np.add(stuff, s)

def fofe_lookahead(state, words, lookahead):
    lookahead_state = []
    for word in words[1:lookahead]:
        fofe_buffer = fofe(lookahead_state, word)
    return fofe_buffer, state


if __name__ == '__main__':
    log.info("INIT")
    nlp = StanfordCoreNLP('http://localhost:9000')
    pre_proc = PreProcessor()
    vocab = {}
    # forget_factor = 0.9

    log.info('VOCAB PASS')
    pathlist = Path("../data/2017").glob('**/*.txt')
    for path in pathlist:
        # because path is object not string
        handle = open(path, "r")
        text = handle.read()
        output_json = nlp.annotate(text, properties={
            'annotators': 'tokenize,ssplit,ner',
            'outputFormat': 'json'
       })


        # init vocabulary
        for sentence in output_json['sentences']:
            for token in sentence['tokens']:
                vocab[pre_proc.tokenGetWord(token)] = 1
                print(pre_proc.tokenGetWord(token))
                #print(token)
                #print('%s' % token['word'])

                #for word in vocab:
                #    print(word)

    # log.info("building one-hot")
    # # build one-hot vectors of floats for each word of the vocab
    # one_hot_length = len(vocab)
    # diag_square_matrix = np.identity(one_hot_length,float)
    # i = 0
    # for word in vocab:
    #     vocab[word] = diag_square_matrix[i]
    #     i = i+1

    log.info('fofe encoding training data')
    fofe = Fofe(vocab, pre_proc, forget_factor=0.7)
    x_train = np.array([]).reshape(0, fofe.getOneHotLength())

    #y_train = keras.utils.to_categorical(np.random.randint(10, size=(1000, 1)), num_classes=10)
    #x_test = np.random.random((100, 100))
    #y_test = keras.utils.to_categorical(np.random.randint(10, size=(100, 1)), num_classes=10)

    pathlist = Path("../data/2017").glob('**/*.txt')
    for path in pathlist:
        # because path is object not string
        handle = open(path, "r")
        text = handle.read()
        output_json = nlp.annotate(text, properties={
            'annotators': 'tokenize,ssplit,ner',
            'outputFormat': 'json'
        })

        left_context = None

        for sentence in output_json['sentences']:
            for token in sentence['tokens']:
                #current_word_vector = vocab[pre_proc.tokenGetWord(token)]
                left_context = fofe.fofe(token)
                #print (x_train, left_context)
                x_train = np.r_[x_train,[left_context]]

        #print(x_train)

        #for row in x_train:
         #   print(row)





    #model = Sequential()

    #model.add(Dense(units=64, input_dim=100))
    #model.add(Activation('relu'))
    #model.add(Dense(units=10))
    #model.add(Activation('softmax'))

    #model.compile(loss='categorical_crossentropy',
    #              optimizer='sgd',
    #              metrics=['accuracy'])




    # x_train and y_train are Numpy arrays --just like in the Scikit-Learn API.
    #model.fit(x_train, y_train, epochs=5, batch_size=32)

    #loss_and_metrics = model.evaluate(x_test, y_test, batch_size=128)
