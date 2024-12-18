from nltk.metrics.association import TOTAL

from BaseTagger import BaseTagger
from collections import defaultdict
from itertools import chain

KNOWN = 0
UNKNOWN = 1
TOTAL = 2

class MLETagger(BaseTagger):
    def __init__(self):
        self.word_tag_counts = defaultdict(lambda: defaultdict(int))

        self.MLE_tags = {}
        self.known_words = 0
        self.unknown_words = 0

    def fit(self, train_set):
        for sent in train_set:
            for word, tag in sent:
                self.word_tag_counts[word][tag] += 1

        for word in self.word_tag_counts:
            self.MLE_tags[word] = max(self.word_tag_counts[word], key=self.word_tag_counts[word].get)

    def predict(self, test_set):
        predictions = {}
        for sent in test_set:
            for word, tag in sent:
                if word in self.MLE_tags:
                    self.known_words += 1
                    predictions[word] = self.MLE_tags[word]
                else:
                    self.unknown_words += 1
                    predictions[word] = "NN"
        return predictions

    def accuracy(self, test_set, predictions):
        test_set = list(chain.from_iterable(test_set))
        total_samples = len(test_set)
        correct_known = sum(1 for word, tag in test_set if word in self.MLE_tags and predictions[word] == tag)
        correct_unknown = sum(1 for word, tag in test_set if word not in self.MLE_tags and predictions[word] == tag)
        correct_total = correct_known + correct_unknown

        known_accuracy = 1 if self.known_words == 0 else correct_known / self.known_words
        unknown_accuracy = 1 if self.unknown_words == 0 else correct_unknown / self.unknown_words
        total_accuracy = 1 if total_samples == 1 else correct_total / total_samples

        return known_accuracy, unknown_accuracy, total_accuracy

    def error_rate(self,test_set):
        predictions = self.predict(test_set)
        model_accuracy = self.accuracy(test_set, predictions)
        return 1 - model_accuracy[KNOWN], 1 - model_accuracy[UNKNOWN], 1 - model_accuracy[TOTAL]