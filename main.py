import nltk
from nltk.corpus import brown
from sklearn.metrics import confusion_matrix

import PseudoCreate

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

def run_model(model, train_set, test_set, model_name):
    print(f"\nRunning {model_name}...")
    model.fit(train_set)
    model_errors_rate = model.error_rate(test_set)
    print(f"{model_name} Error Rates:")
    print(f"Known: {model_errors_rate[0]:.4f}, Unknown: {model_errors_rate[1]:.4f}, Total: {model_errors_rate[2]:.4f}")
    return model_errors_rate, model

def compare_models(comparison_title, first_model_error_rates, second_model_error_rates):
    print("\n" + comparison_title)
    print(f"Known Words Improvement: {first_model_error_rates[0] - second_model_error_rates[0]:.4f}")
    print(f"Unknown Words Improvement: {first_model_error_rates[1] - second_model_error_rates[1]:.4f}")
    print(f"Total Words Improvement: {first_model_error_rates[2] - second_model_error_rates[2]:.4f}")


if __name__ == '__main__':
    # task A

    # Load training and testing data
    train_set, test_set = load_data()

    # task B
    mle_error_rates = run_model(MLETagger(), train_set, test_set, "MLE Tagger")[0]

    # Task C

    # Bigram HMM Tagger
    hmm_error_rates = run_model(BIgramHMMTagger(), train_set, test_set, "Bigram HMM Tagger")[0]

    # Comparing C and B
    compare_models("MLE and HMM comparison:", mle_error_rates, hmm_error_rates)

    # Task D

    # Add One Smoothing
    smoothed_hmm_error_rates = run_model(BIgramHMMTagger(apply_smoothing=True), train_set, test_set, "Bigram HMM Tagger With Smoothing")[0]

    # Comparing B and D
    compare_models("MLE and Smoothing HMM comparison:", mle_error_rates, smoothed_hmm_error_rates)

    # Comparing C and D
    compare_models("HMM and Smoothing HMM comparison:", hmm_error_rates, smoothed_hmm_error_rates)

    # Task E

    #Pseudo Words
    pseudo_map = PseudoCreate.pseudo_set(train_set)
    with_pseudo_error_rates = run_model(BIgramHMMTagger(pseudo_map), train_set, test_set, "Bigram HMM Tagger With Pseudo Words")[0]

    # Comparing B and E1
    compare_models("MLE and Pesudo HMM comparison:", mle_error_rates, with_pseudo_error_rates)

    # Comparing C and E1
    compare_models("HMM and Pesudo HMM comparison:", hmm_error_rates, with_pseudo_error_rates)

    # Comparing D and E1
    compare_models("Smoothing HMM and Pesudo HMM comparison:", smoothed_hmm_error_rates, with_pseudo_error_rates)

    # Add One Smoothing
    pseudo_map = PseudoCreate.pseudo_set(train_set)
    with_pseudo_smoothed_error_rates, with_pseudo_smoothed_bigram_hmm_tagger = run_model(BIgramHMMTagger(pseudo_map, apply_smoothing=True), train_set, test_set, "Bigram HMM Tagger With Pseudo Words + Smoothing")

    # Comparing B and E2
    compare_models("MLE and Pseudo + Smoothing HMM comparison:", mle_error_rates, with_pseudo_smoothed_error_rates)

    # Comparing C and E2
    compare_models("HMM and Pseudo + Smoothing HMM comparison:", hmm_error_rates, with_pseudo_smoothed_error_rates)

    # Comparing D and E2
    compare_models("Smoothing HMM and Pseudo + Smoothing comparison:", smoothed_hmm_error_rates,
                   with_pseudo_smoothed_error_rates)

    # Comparing E1 and E2
    compare_models("Pseudo HMM and Pseudo + Smoothing HMM comparison:", with_pseudo_error_rates,
                   with_pseudo_smoothed_error_rates)

    # Confusion Matrix
    confusion_matrix = with_pseudo_smoothed_bigram_hmm_tagger.create_confusion_matrix(test_set)
    top_errors = with_pseudo_smoothed_bigram_hmm_tagger.top_confusion_errors(confusion_matrix, top_n=10)
    print("\nTop 10 Errors from Confusion Matrix Are: ")
    print(*top_errors, sep="\n")



