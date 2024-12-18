from BaseTagger import BaseTagger
from collections import defaultdict

START_WORD = "<s>"
START_TAG = "<START>"
END_TAG = "<END>"

class BIgramHMMTagger(BaseTagger):
    def __init__(self):
        super()
        self.bigram_tag_counter = defaultdict(lambda :defaultdict(int))
        self.probabilites = defaultdict(lambda :defaultdict(float))
        self.emissions = defaultdict(lambda :defaultdict(float))
        self.emissions[START_TAG][START_WORD] = 1.0

        self.tag_word_counter = defaultdict(lambda :defaultdict(int))
        self.tags = set()
        self.known_words = set()

    def fit(self, train_set):
        self.tags.add(START_TAG)
        for sentence in train_set:
            prev_tag = START_TAG
            for word, tag in sentence:
                self.known_words.add(word)
                self.tags.add(tag)

                self.bigram_tag_counter[prev_tag][tag] += 1
                self.tag_word_counter[tag][word] += 1
                prev_tag = tag

            self.bigram_tag_counter[prev_tag][END_TAG] += 1
        self.tags.add(END_TAG)

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

    def predict(self, test_set):
        predictions = []
        for sentence in test_set:
            sentence.insert(0, (START_WORD, START_TAG))
            n = len(sentence)
            viterbi = defaultdict(lambda :defaultdict(float))
            backpointer = defaultdict(lambda :defaultdict(str))

            viterbi[0][START_TAG] = 1.0

            for t, word_tag in enumerate(sentence):
                word = word_tag[0]
                for tag in self.tags:
                    max_prob = 0
                    best_prev_tag = None
                    for prev_tag in self.tags:
                        transition_prob = self.probabilites[prev_tag][tag]
                        emission_prob = self.emissions[tag][word]
                        prob = viterbi[t][prev_tag] * transition_prob * emission_prob
                        if prob > max_prob:
                            max_prob = prob
                            best_prev_tag = prev_tag
                    viterbi[t+1][tag] = max_prob
                    backpointer[t+1][tag] = best_prev_tag

            last_tag = max(viterbi[n], key=viterbi[n].get)
            best_tags = [last_tag]

            for t in range(n, 0, -1):
                best_tags.insert(0, backpointer[t][last_tag])
                last_tag = backpointer[t][last_tag]

            predictions.append(list(zip(sentence, best_tags[1:])))

        return predictions

    def accuracy(self, test_set, predictions):
        total_known = 0
        total_unknown = 0
        correct_known = 0
        correct_unknown = 0

        for sentence, prediction in zip(test_set, predictions):
            sentence.insert(0, (START_WORD, START_TAG))
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

    # def error_rate(self,test_set):
    #     predictions = self.predict(test_set)
    #     model_accuracy = self.accuracy(test_set, predictions)
    #     return