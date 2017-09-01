class PreProcessor:
    # if the token is $100.0 or some number
    # then replace it with NUMBER
    def tokenGetWord(self, token):
        if token['ner'] in set(['MONEY', 'NUMBER']):
            return token['ner']
        else:
            return token['word']