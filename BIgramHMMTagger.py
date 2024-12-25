from BaseTagger import BaseTagger, UNKNOWN
from collections import defaultdict

START_WORD = "<s>"
START_TAG = "<START>"
END_TAG = "<END>"
UNKNOWN_TAG = "UNKNOWN"


class BIgramHMMTagger(BaseTagger):
    def __init__(self):
        super()
        self.bigram_tag_counter = defaultdict(lambda: defaultdict(int))
        self.probabilites = defaultdict(lambda: defaultdict(float))
        self.emissions = defaultdict(lambda: defaultdict(float))
        self.emissions[START_TAG][START_WORD] = 1.0

        self.tag_word_counter = defaultdict(lambda: defaultdict(int))
        self.tags = {START_TAG, END_TAG, UNKNOWN_TAG}
        self.known_words = set()

    def fit(self, train_set):
        """
        Compute transition and emission probabilities from the training set.
        :param train_set:
        """
        for sentence in train_set:
            prev_tag = START_TAG
            for word, tag in sentence:
                # Add the word to the known words
                self.known_words.add(word)
                self.tags.add(tag)

                # Count bigram transitions
                self.bigram_tag_counter[prev_tag][tag] += 1

                # Count tag-word emissions
                self.tag_word_counter[tag][word] += 1

                # Update the previous tag
                prev_tag = tag

            # Transition from the last tag to END_TAG
            self.bigram_tag_counter[prev_tag][END_TAG] += 1

        # Compute transition probabilities
        for prev_tag, next_tags in self.bigram_tag_counter.items():
            total_count = sum(next_tags.values())
            for next_tag, count in next_tags.items():
                self.probabilites[prev_tag][next_tag] = count / total_count

        # Compute emission probabilities
        for tag, words in self.tag_word_counter.items():
            total_count = sum(words.values())
            for word, count in words.items():
                self.emissions[tag][word] = count / total_count

    def viterbi(self, sentence):
        n = len(sentence)
        viterbi_table = defaultdict(lambda: defaultdict(float))
        back_pointer = defaultdict(lambda: defaultdict(str))

        # Initialization step
        viterbi_table[0][START_TAG] = 1.0

        # Recursion step
        for t in range(1, n + 1):
            word = sentence[t - 1][0]
            for current_tag in self.tags:
                max_prob, best_prev_tag = 0.0, None
                for prev_tag in self.tags:
                    prob = (
                            viterbi_table[t - 1][prev_tag]
                            * self.probabilites[prev_tag][current_tag]
                            * (self.emissions[current_tag][word] if word in self.known_words else 1.0)
                    )
                    if prob > max_prob:
                        max_prob = prob
                        best_prev_tag = prev_tag

                # Update Viterbi table and back_pointer
                viterbi_table[t][current_tag] = max_prob
                back_pointer[t][current_tag] = best_prev_tag

        # Termination step
        best_last_tag = max(viterbi_table[n], key=viterbi_table[n].get)
        best_path = [best_last_tag]

        # Backtracking
        for t in range(n, 0, -1):
            best_path.insert(0, back_pointer[t][best_path[0]])

        return best_path[1:]  # Exclude START_TAG from the result

    def predict(self, test_set):
        predictions = []
        for sentence in test_set:
            best_tags = self.viterbi(sentence)
            predictions.append(list(zip([word for word, _ in sentence], best_tags)))
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

        correct_total = correct_known + correct_unknown
        total = total_known + total_unknown

        known_accuracy = correct_known / total_known
        unknown_accuracy = correct_unknown / total_unknown
        total_accuracy = correct_total / total

        return known_accuracy, unknown_accuracy, total_accuracy
