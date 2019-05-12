import pytest
import os
import shutil
import pickle as pkl

from quadboost.callbacks import ModelCheckpoint, CallbacksManagerIterator


class DummyModel:
    def __init__(self, a=1):
        self.a = a
        self.weak_predictors = []
        self.weak_predictors_weights = []

    def __eq__(self, other):
        return self.__dict__ == other.__dict__


class TestModelCheckpoint:
    def setup_method(self, method):
        self.model = DummyModel(42)
        self.manager = CallbacksManagerIterator(self.model)
        self.ckpt = ModelCheckpoint(filename='test_ckpt', dirname='./test_ckpt', manager=self.manager)

    def teardown_method(self, method):
        if os.path.isdir('./test_ckpt'):
            shutil.rmtree('./test_ckpt')

    def test_create_directory(self):
        assert os.path.isdir("./test_ckpt")

    def test_filedir_correctly_formated(self):
        assert os.path.normpath(self.ckpt.filedir) == os.path.normpath('./test_ckpt/test_ckpt')

    def test_filedir_correctly_formated_with_round_number(self):
        self.ckpt.filename = 'test_ckpt_{round}'
        assert os.path.normpath(self.ckpt.filedir) == os.path.normpath('./test_ckpt/test_ckpt_0')

    def test_initial_save_on_iteration_begin(self):
        self.ckpt.on_iteration_begin() # Open file and save
        self.ckpt.on_iteration_end() # Closes file
        assert os.path.exists('./test_ckpt/test_ckpt.pkl')
        with open('./test_ckpt/test_ckpt.pkl', 'rb') as file:
            model = pkl.load(file)
        assert model == self.model

    def test_save_on_step_end(self):
        self.ckpt.on_iteration_begin() # Open file and save
        # Update model
        self.model.weak_predictors.append('first weak predictor')
        self.model.weak_predictors_weights.append('first weak predictor weight')
        self.ckpt.on_step_end() # Saves update

        self.model.weak_predictors.append('second weak predictor')
        self.model.weak_predictors_weights.append('second weak predictor weight')
        self.ckpt.on_step_end() # Saves update

        self.ckpt.on_iteration_end() # Closes file

        with open('./test_ckpt/test_ckpt.pkl', 'rb') as file:
            model = pkl.load(file)

            (wp, wpw) = pkl.load(file)
            model.weak_predictors.append(wp)
            model.weak_predictors_weights.append(wpw)

            (wp, wpw) = pkl.load(file)
            model.weak_predictors.append(wp)
            model.weak_predictors_weights.append(wpw)
        assert model == self.manager.caller


