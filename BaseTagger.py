from abc import abstractmethod



KNOWN = 0
UNKNOWN = 1
TOTAL = 2

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

    def error_rate(self,test_set):
        predictions = self.predict(test_set)
        model_accuracy = self.accuracy(test_set, predictions)
        return 1 - model_accuracy[KNOWN] , 1 - model_accuracy[UNKNOWN] , 1 - model_accuracy[TOTAL]
