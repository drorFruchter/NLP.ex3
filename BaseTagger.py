from abc import abstractmethod


class BaseTagger(object):
    @abstractmethod
    def fit(self, train_set):
        pass

    @abstractmethod
    def predict(self, test_set):
        pass

    @abstractmethod
    def accuracy(self, test_set, predictions):
        pass

    @abstractmethod
    def error_rate(self,test_set):
        pass
