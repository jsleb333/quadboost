from sklearn.metrics import accuracy_score
import inspect


class _Cloner:
    """
    This class is essentially a decorator around the constructor of its subclasses, but without the inconveniences of decorators (with pickling).

    The class becomes clonable with the same init args and kwargs when the object is called. This allows to use instanciated objects as if they were the class itself.
    """
    def __new__(cls, *args, **kwargs):
        """
        This constructor class makes any class clonable by setting the __call__ function as a constructor using the initialization parameters.
        """
        def clone(self):
            return cls(*self._init_args, **self._init_kwargs)
        cls.__call__ = clone

        new_weak_learner = super().__new__(cls)

        new_weak_learner._init_args = args
        new_weak_learner._init_kwargs = kwargs

        return new_weak_learner


class _WeakLearnerBase(_Cloner):
    """
    This class implements a abstract base weak learner that should be inherited. It makes sure all weak learner are clonable and have an encoder, as well as a predict and an evaluate methods.
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
