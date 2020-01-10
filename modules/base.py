import numpy as np
import torch
from torch import nn
import torch.optim as optim
from torch.autograd import Variable


FRAMEWORKS = ['tf', 'torch']
_DEFAULT_CONFIG = {
    'batch_size': 32,
    'input_size': 1512,
    'hidden_size': 128,
    'projection_size': 200,
    'learning_rate': 1e-3,
    'grad_clip': False,
    'grad_clip_norm': 2.0,
    '_lambda': 0.0,
    'p_keep_input': 0.9,
    'p_keep_rnn': 0.9,
    'p_keep_fc': 0.9,
    'num_classes': 7,
    'enc_type': 'rnn'
}


class Config:

    def __init__(self, default=_DEFAULT_CONFIG, **kwargs):
        self.default = default
        self.kwargs = kwargs
        self.batch_size = self._value('batch_size', kwargs)
        self.input_size = self._value('input_size', kwargs)
        self.hidden_size = self._value('hidden_size', kwargs)
        self.projection_size = self._value('projection_size', kwargs)
        self.learning_rate = self._value('learning_rate', kwargs)
        self.grad_clip = self._value('grad_clip', kwargs)
        self.grad_clip_norm = self._value('grad_clip_norm', kwargs)
        self._lambda = self._value('_lambda', kwargs)
        self.p_keep_input = self._value('p_keep_input', kwargs)
        self.p_keep_rnn = self._value('p_keep_rnn', kwargs)
        self.p_keep_fc = self._value('p_keep_fc', kwargs)
        self.enc_type = self._value('enc_type', kwargs)
        self.num_classes = self._value('num_classes', kwargs)

        for key in [k for k in kwargs.keys()
                    if k not in self.default.keys()]:
            setattr(self, key, kwargs[key])

    def __delitem__(self, key):
        pass

    def __getitem__(self, key):
        return self.__getattribute__(key)

    def __repr__(self):
        x = 'Config as follows:\n'
        for key in sorted(self.keys()):
            x += '\t%s \t%s%s\n' % \
                 (key, '\t' if len(key) < 15 else '', self[key])
        return x

    def __setitem__(self, key, value):
        self.__setattr__(key, value)

    def dropout_keys(self):
        return [k for k in self.__dict__.keys() if k.startswith('p_keep_')]

    def keys(self):
        return [key for key in self.__dict__.keys()
                if key not in ['default', 'kwargs']]

    def to_json(self):
        return dict(self.__dict__)

    def _value(self, key, kwargs):
        if key in kwargs.keys():
            return kwargs[key]
        else:
            return self.default[key]


class Model:

    def __init__(self, framework, config):
        self.framework = framework
        self.config = config
        for key in config.keys():
            setattr(self, key, config[key])

    def forward(self, *args):
        raise NotImplementedError

    def loss(self, *args):
        raise NotImplementedError

    def optimize(self, *args):
        raise NotImplementedError


class PyTorchModel(Model, nn.Module):

    def __init__(self, name, config):
        Model.__init__(self, 'pytorch', config)
        nn.Module.__init__(self)
        self.name = name

    def forward(self, forest):
        raise NotImplementedError

    def loss(self, predictions, targets):
        loss = self.criterion(predictions, targets)
        return loss

    def optimize(self, loss):
        loss.backward()
        if self.grad_clip:
            torch.nn.utils.clip_grad_norm(
                self.parameters(), self.grad_clip_norm)
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()
