from abc import abstractmethod


class BaseModel:

    @abstractmethod
    def fit(self,  X_train, y_train):
        """
        Fit the model to training data.
        """
        return

    @abstractmethod
    def predict(self, X_test):
        """
        Predictions with the model. Returns posterior means and standard deviations at X. Note that this is different in GPy where the variances are given.
        """
        return