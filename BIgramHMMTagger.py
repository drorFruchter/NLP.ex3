import numpy as np
import PseudoCreate
from BaseTagger import BaseTagger, UNKNOWN
from collections import defaultdict

START_WORD = "<s>"
START_TAG = "<START>"
END_TAG = "<END>"
UNKNOWN_TAG = "NN"


class BIgramHMMTagger(BaseTagger):
    def __init__(self, pseudo_map=None, apply_smoothing=False):
        super()
        self.bigram_tag_counter = defaultdict(lambda: defaultdict(int))
        self.probabilites = defaultdict(lambda: defaultdict(float))
        self.emissions = defaultdict(lambda: defaultdict(float))
        self.emissions[START_TAG][START_WORD] = 1.0

        self.tag_word_counter = defaultdict(lambda: defaultdict(int))
        self.tags = {START_TAG, END_TAG, UNKNOWN_TAG}
        self.known_words = set()
        self.pseudo_map = pseudo_map
        self.apply_smoothing = apply_smoothing
        self.predictions = None

    def fit(self, train_set):
        for sentence in train_set:
            prev_tag = START_TAG
            for word, tag in sentence:
                if self.pseudo_map is not None and word in self.pseudo_map:
                    word = self.pseudo_map[word]
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
        vocab_size = len(self.known_words)  # Total unique words in the training set
        for tag, words in self.tag_word_counter.items():
            total_count = sum(words.values())
            for word in self.known_words:
                if self.apply_smoothing:
                    smoothed_count = self.tag_word_counter[tag][word] + 1
                    smoothed_total = total_count + vocab_size
                    self.emissions[tag][word] = smoothed_count / smoothed_total
                else:
                    self.emissions[tag][word] = (
                            self.tag_word_counter[tag][word] / total_count) if total_count > 0 else 0.0

    def viterbi(self, sentence):
        n = len(sentence)
        viterbi_table = defaultdict(lambda: defaultdict(float))
        back_pointer = defaultdict(lambda: defaultdict(str))

        # Initialization step
        viterbi_table[0][START_TAG] = 1.0

        # Recursion step
        for t in range(1, n + 1):
            word = sentence[t - 1][0]
            if self.pseudo_map is not None and word not in self.pseudo_map.values():
                word = PseudoCreate.create_pseudo(word)
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
        self.predictions = predictions
        return predictions

    def accuracy(self, test_set, predictions):
        total_known = 0
        total_unknown = 0
        correct_known = 0
        correct_unknown = 0

        for sentence, prediction in zip(test_set, predictions):
            for (word, true_tag), (_, predicted_tag) in zip(sentence, prediction):
                if self.pseudo_map is not None and word in self.pseudo_map:
                    word = self.pseudo_map[word]

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



    def create_confusion_matrix(self, test_set):
        tag_to_idx = {tag: idx for idx, tag in enumerate(self.tags)}
        n_tags = len(self.tags)

        confusion_matrix = np.zeros((n_tags, n_tags))
        for sentence, pred_sentence in zip(test_set, self.predictions):
            for (_, true_tag), (_, predicted_tag) in zip(sentence, pred_sentence):
                if true_tag in tag_to_idx and predicted_tag in tag_to_idx:
                    true_idx = tag_to_idx[true_tag]
                    predicted_idx = tag_to_idx[predicted_tag]
                    confusion_matrix[true_idx, predicted_idx] += 1
        return confusion_matrix

    def top_confusion_errors(self, confusion_matrix, top_n):
        tag_to_idx = {tag: idx for idx, tag in enumerate(self.tags)}
        idx_to_tag = {idx: tag for tag, idx in tag_to_idx.items()}

        errors = []
        n_tags = len(self.tags)

        for i in range(n_tags):
            for j in range(n_tags):
                if i != j:
                    count = confusion_matrix[i, j]
                    if count > 0:
                        errors.append((int(count), {"true_tag": idx_to_tag[i],"predicted_tag": idx_to_tag[j]}))

        errors.sort(key=lambda x: x[0], reverse=True)
        return errors[:top_n]
