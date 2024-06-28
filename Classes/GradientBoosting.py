from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import GradientBoostingRegressor


class GradientBoosting(BaseEstimator, RegressorMixin):
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.model = GradientBoostingRegressor(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth
        )

    def fit(self, x, y):  # x = feature matrix, y = target vector
        # train the gradient boosting model on dataset provided
        self.model.fit(x, y)

    def predict(self, x):  # x = feature matrix
        # generate predictions for x using trained gradient boosting model
        return self.model.predict(x)

    def get_params(self, deep=True):
        # return hyperparameters as a dictionary
        return {
            'n_estimators': self.n_estimators,
            'learning_rate': self.learning_rate,
            'max_depth': self.max_depth
        }

    def set_params(self, **params):
        # set hyperparameters from a dictionary
        for key, value in params.items():
            setattr(self, key, value)
        # reinitialize the model with updated hyperparameters
        self.model = GradientBoostingRegressor(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth
        )
        return self
