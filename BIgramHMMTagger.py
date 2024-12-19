from BaseTagger import BaseTagger, UNKNOWN
from collections import defaultdict

START_WORD = "<s>"
START_TAG = "<START>"
END_TAG = "<END>"
UNKNOWN_TAG = "UNKNOWN"

class BIgramHMMTagger(BaseTagger):
    def __init__(self):
        super()
        self.bigram_tag_counter = defaultdict(lambda :defaultdict(int))
        self.probabilites = defaultdict(lambda :defaultdict(float))
        self.emissions = defaultdict(lambda :defaultdict(float))
        self.emissions[START_TAG][START_WORD] = 1.0

        self.tag_word_counter = defaultdict(lambda :defaultdict(int))
        self.tags = {START_TAG, END_TAG, UNKNOWN_TAG}
        self.known_words = set()

    def fit(self, train_set):
        for sentence in train_set:
            prev_tag = START_TAG
            for word, tag in sentence:
                self.known_words.add(word)
                self.tags.add(tag)

                self.bigram_tag_counter[prev_tag][tag] += 1
                self.tag_word_counter[tag][word] += 1
                prev_tag = tag

            self.bigram_tag_counter[prev_tag][END_TAG] += 1

        #probabilites
        for prev_tag, next_tags in self.bigram_tag_counter.items():
            total_count = sum(next_tags.values())
            for next_tag, count in next_tags.items():
                self.probabilites[prev_tag][next_tag] = count / total_count

        #emissions
        for tag, words in self.tag_word_counter.items():
            total_count = sum(words.values())
            for word, count in words.items():
                self.emissions[tag][word] = count / total_count

    def viterbi(self, sentence):
        n = len(sentence)
        viterbi_table = defaultdict(lambda :defaultdict(float))
        backpointer = defaultdict(lambda :defaultdict(str))

        viterbi_table[0][START_TAG] = 1.0

        for t in range(1, n + 1):
            word = sentence[t-1][0]
            for tag in self.tags:
                 max_prob = max(viterbi_table[t-1][prev_tag] * self.probabilites[prev_tag][tag] for prev_tag in self.tags)
                 viterbi_table[t][tag] = max_prob * (self.emissions[tag][word] if word in self.known_words else 1.0)
                 backpointer[t][tag] = max(self.tags, key=lambda prev_tag: viterbi_table[t-1][prev_tag] * self.probabilites[prev_tag][tag])

        best_last_tag = max(viterbi_table[n], key=viterbi_table[n].get)
        best_path = [best_last_tag]

        for t in range(n, 0, -1):
            best_path.insert(0, backpointer[t][best_path[0]])

        return best_path

    def predict(self, test_set):
        predictions = []
        for sentence in test_set:
            best_tags = self.viterbi(sentence)
            predictions.append(list(zip([word for word, _ in sentence], best_tags[1:])))
        return predictions

    def accuracy(self, test_set, predictions):
        total_known = 0
        total_unknown = 0
        correct_known = 0
        correct_unknown = 0

        for sentence, prediction in zip(test_set, predictions):
            for (word, true_tag), (_, predicted_tag) in zip(sentence, prediction):
                if word in self.known_words:
                    total_known += 1
                    if true_tag == predicted_tag:
                        correct_known += 1
                else:
                    total_unknown += 1
                    if true_tag == predicted_tag:
                        correct_unknown += 1

        correct_total =  correct_known + correct_unknown
        total = total_known + total_unknown

        known_accuracy = correct_known / total_known
        unknown_accuracy = correct_unknown / total_unknown
        total_accuracy = correct_total / total

        return known_accuracy, unknown_accuracy, total_accuracy