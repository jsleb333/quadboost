import pytest
import pickle as pkl
import os
from quadboost.weak_learner import _Cloner


class SomeClonableClass(_Cloner):
    def __init__(self, some_args, some_kwargs='stock kwargs'):
        self.some_args = some_args
        self.some_kwargs = some_kwargs


class TestCloner:
    def setup_method(self):
        self.args, self.kwargs = (1,), {'some_kwargs':'not stock kwargs'}
        self.some_clonable_obj = SomeClonableClass(*self.args, **self.kwargs)

    def test_clone_obj(self):
        clone = self.some_clonable_obj()
        assert clone.some_args ==  self.args[0]
        assert clone.some_kwargs ==  self.kwargs['some_kwargs']
        assert clone is not self.some_clonable_obj

    def test_pickled_obj_still_clonable(self):
        filepath = 'tests/weak_learner/cloned.pkl'
        with open(filepath, 'wb') as file:
            pkl.dump(self.some_clonable_obj, file)

        with open(filepath, 'rb') as file:
            unpickled_clonable_obj = pkl.load(file)

        clone = unpickled_clonable_obj()
        assert clone.some_args ==  self.args[0]
        assert clone.some_kwargs ==  self.kwargs['some_kwargs']
        assert clone is not unpickled_clonable_obj

        os.remove(filepath)


class Test_WeakLearnerBase:
    pass
