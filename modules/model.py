import torch
from torch import nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F

from modules import base, encoder


class ClonalModel(base.PyTorchModel):
    def __init__(self, name, config):
        super(ClonalModel, self).__init__(name, config)

        if self.enc_type == "rnn":
            self.encoder = encoder.ChildSumTreeLSTMRNNEncoder(
                self.input_size, self.hidden_size,
                self.p_keep_input, self.p_keep_rnn)
        elif self.enc_type == "mean":
            self.encoder = encoder.ChildSumTreeLSTMMeanEncoder(
                self.input_size, self.hidden_size,
                self.p_keep_input, self.p_keep_rnn)

    @staticmethod
    def current_batch_size(forest):
        return forest.num_trees

    def forward(self, forest):
        root_states = self.encoder.forward(forest)
        root_states = torch.stack([r[1] for r in root_states])

        if root_states.size()[1] == 2:
            print(root_states.size()[1])
            root_states = F.torch.sum(root_states, 1, keepdim=True)

        linear = self.linear(root_states.view(-1, self.hidden_size))
        return self.compute_output(linear, forest)


class ClonalModelClassifier(ClonalModel):

    def __init__(self, name, config):
        super(ClonalModelClassifier, self).__init__(name, config)
        self.criterion = nn.CrossEntropyLoss()
        self.linear = nn.Linear(self.hidden_size, self.num_classes)
        self.softmax = nn.Softmax(dim=1)
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

    def compute_output(self, linear, forest):
        labels = Variable(torch.LongTensor(forest.labels()),
                          requires_grad=False)
        softmax = self.softmax(linear)
        loss = self.loss(softmax, labels)
        predictions = self.predictions(softmax).type_as(labels)
        correct = self.correct_predictions(predictions, labels)
        accuracy = self.accuracy(correct, forest.num_trees)
        return predictions, loss, accuracy.item(0)

    @staticmethod
    def accuracy(correct_predictions, batch_size):
        correct = correct_predictions.cpu().sum().data.numpy()
        return correct / float(batch_size)

    @staticmethod
    def correct_predictions(predictions, labels):
        return predictions.eq(labels)

    @staticmethod
    def predictions(logits):
        return logits.max(1)[1]


class ClonalModelRegressor(ClonalModel):
    def __init__(self, name, config):
        super(ClonalModelRegressor, self).__init__(name, config)
        self.criterion = nn.MSELoss()
        self.linear = nn.Linear(self.hidden_size, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

    def compute_output(self, linear, forest):
        target = Variable(torch.Tensor(forest.labels()),
                          requires_grad=False)
        return self.loss(linear, target)
