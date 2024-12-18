import nltk
from nltk.corpus import brown
nltk.download('brown')
from sklearn.model_selection import train_test_split
from MLETagger import MLETagger
from BIgramHMMTagger import BIgramHMMTagger


def load_data():
    dataset = brown.tagged_sents(categories='news')
    def simplify_tag(tag):
        return tag.split('+')[0].split('-')[0]

    dataset = [[(word, simplify_tag(tag)) for word, tag in sent] for sent in dataset]
    train_set, test_set = train_test_split(dataset, test_size=0.1)
    return train_set, test_set


if __name__ == '__main__':
    train_set, test_set = load_data()

    #a
    # MLE_model = MLETagger()
    # MLE_model.fit(train_set)
    # print(MLE_model.error_rate(test_set))

    #b
    BIgramHMMTagger = BIgramHMMTagger()
    BIgramHMMTagger.fit(train_set)
    print(BIgramHMMTagger.error_rate(test_set))