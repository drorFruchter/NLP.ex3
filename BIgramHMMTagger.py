from BaseTagger import BaseTagger

class BIgramHMMTagger(BaseTagger):
    def __init__(self, tagger):
        self.tagger = tagger

    def fit(self, train_set):
        pass