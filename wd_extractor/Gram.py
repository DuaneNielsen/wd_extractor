from intervals import IntInterval

class Gram:

    def __init__(self, document, interval, named_entity):
        self.document = document
        self.interval = interval
        self.named_entity = named_entity

    def vector(self):
        return None

    def output(self):
        return None