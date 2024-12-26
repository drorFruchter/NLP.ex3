import nltk
from nltk.corpus import brown

nltk.download('brown')
from sklearn.model_selection import train_test_split
from MLETagger import MLETagger
from BIgramHMMTagger import BIgramHMMTagger


def load_data():
    dataset = brown.tagged_sents(categories='news')

    def simplify_tag(tag):
        return tag.split('+')[0].split('-')[0].replace('$', '').replace('*', '')

    dataset = [[(word, simplify_tag(tag)) for word, tag in sent] for sent in dataset]
    train_set, test_set = train_test_split(dataset, test_size=0.1)
    return train_set, test_set


if __name__ == '__main__':
    # task A

    # Load training and testing data
    train_set, test_set = load_data()

    # task B

    MLE_model = MLETagger()
    MLE_model.fit(train_set)
    mle_error_rates = MLE_model.error_rate(test_set)
    print("MLE_model Tagger Error Rates:")
    print(f"Known: {mle_error_rates[0]:.4f}, Unknown: {mle_error_rates[1]:.4f}, Total: {mle_error_rates[2]:.4f}")

    # Task C

    # Bigram HMM Tagger
    print("\nRunning Bigram HMM Tagger...")
    bigram_hmm_tagger = BIgramHMMTagger()
    bigram_hmm_tagger.fit(train_set)
    hmm_predictions = bigram_hmm_tagger.predict(test_set)
    hmm_error_rates = bigram_hmm_tagger.error_rate(test_set)
    print("Bigram HMM Tagger Error Rates:")
    print(f"Known: {hmm_error_rates[0]:.4f}, Unknown: {hmm_error_rates[1]:.4f}, Total: {hmm_error_rates[2]:.4f}")

    # Task D
    smoothed_hmm_tagger = BIgramHMMTagger()
    smoothed_hmm_tagger.fit(train_set, apply_smoothing=True)
    smoothed_hmm_error_rates = smoothed_hmm_tagger.error_rate(test_set)
    print("Bigram HMM Tagger Error Rates:")
    print(f"Known: {smoothed_hmm_error_rates[0]:.4f}, Unknown: {smoothed_hmm_error_rates[1]:.4f},"
          f" Total: {smoothed_hmm_error_rates[2]:.4f}")

    # Task E
    

    # Compare Results
    print("\nComparison of Error Rates:")
    # Comparing B and C
    print(f"Known Words Improvement: {mle_error_rates[0] - hmm_error_rates[0]:.4f}")
    print(f"Unknown Words Improvement: {mle_error_rates[1] - hmm_error_rates[1]:.4f}")
    print(f"Total Words Improvement: {mle_error_rates[2] - hmm_error_rates[2]:.4f}")

    # Comparing B and D
    print(f"Known Words Improvement: {mle_error_rates[0] - smoothed_hmm_error_rates[0]:.4f}")
    print(f"Unknown Words Improvement: {mle_error_rates[1] - smoothed_hmm_error_rates[1]:.4f}")
    print(f"Total Words Improvement: {mle_error_rates[2] - smoothed_hmm_error_rates[2]:.4f}")

    # Comparing C and D
    print(f"Known Words Improvement: {hmm_error_rates[0] - smoothed_hmm_error_rates[0]:.4f}")
    print(f"Unknown Words Improvement: {hmm_error_rates[1] - smoothed_hmm_error_rates[1]:.4f}")
    print(f"Total Words Improvement: {hmm_error_rates[2] - smoothed_hmm_error_rates[2]:.4f}")

