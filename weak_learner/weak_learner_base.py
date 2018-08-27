from sklearn.metrics import accuracy_score
import inspect


class Cloner:
    """
    This constructor class makes any weak learners clonable by setting the __call__ function as a constructor using the initialization parameters.
    This class, to be inherited, acts like a decorator around the constructor, but without the inconveniences of decorators (with pickling).
    """
    def __new__(cls, *args, **kwargs):
        def clone(self): return cls(*args, **kwargs)
        cls.__call__ = clone
        return super().__new__(cls)


class WeakLearnerBase(Cloner):
    """
    This class implements a abstract base weak learner that should be inherited. It makes sure all weak learner have an encoder and an evaluate method.
    """
    def __init__(self, *args, encoder=None, **kwargs):
        self.encoder = encoder
        super().__init__(*args, **kwargs)

    def fit(self, X, Y, W=None, **kwargs):
        raise NotImplementedError

    def predict(self, X, Y):
        raise NotImplementedError

    def evaluate(self, X, Y):
        Y_pred = self.predict(X)
        if self.encoder != None:
            Y_pred = self.encoder.decode_labels(Y_pred)
        return accuracy_score(y_true=Y, y_pred=Y_pred)