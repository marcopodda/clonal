import torch
import torch.nn as nn
from torch.autograd import Variable

from modules import cell


class RNNEncoder(nn.Module):
    def __init__(self, input_size, hidden_size=1000):
        super(RNNEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size, 1)

    def forward(self, input, hidden):
        output, hidden_state = self.gru(input, hidden)
        return output, hidden_state

    def first_hidden(self):
        return Variable(torch.FloatTensor(1, 1, self.hidden_size).zero_())


class ChildSumTreeLSTMEncoder(nn.Module):
    def __init__(self, input_size, hidden_size,
                 p_keep_input, p_keep_rnn):
        super(ChildSumTreeLSTMEncoder, self).__init__()
        self._drop_input = nn.Dropout(p=1.0 - p_keep_input)

        self.input_size = input_size
        self.hidden_size = hidden_size

    def _get_input(self, vec):
        return Variable(torch.Tensor(vec).expand(1, self.input_size), requires_grad=False)

    def _get_hidden(self, children):
        num_children = len(children)

        if num_children == 0:
            child_c = Variable(torch.zeros(1, 1, self.hidden_size))
            child_h = Variable(torch.zeros(1, 1, self.hidden_size))
        else:
            child_c = Variable(torch.Tensor(num_children, 1, self.hidden_size))
            child_h = Variable(torch.Tensor(num_children, 1, self.hidden_size))

            for idx in range(num_children):
                child_c[idx], child_h[idx] = children[idx].state

        return child_c, child_h


class ChildSumTreeLSTMRNNEncoder(ChildSumTreeLSTMEncoder):

    def __init__(self, input_size, hidden_size,
                 p_keep_input, p_keep_rnn):
        super(ChildSumTreeLSTMRNNEncoder, self).__init__(input_size, hidden_size,
                                                         p_keep_input, p_keep_rnn)
        self.rnn = RNNEncoder(input_size, hidden_size)
        self.cell = cell.BatchChildSumTreeLSTMCell(
            input_size=hidden_size,
            hidden_size=hidden_size,
            p_dropout=1.0 - p_keep_rnn)

    def forward(self, forest):
        outputs = []
        for tree in forest:
            for node in tree:
                encodings = node.encoding
                hidden_rnn = self.rnn.first_hidden()
                for enc in encodings:
                    enc = self._get_input(enc).expand(1, 1, self.input_size)
                    _, hidden_rnn = self.rnn.forward(enc, hidden_rnn)
                node_encoding = hidden_rnn.view(1, -1)
                hidden = self._get_hidden(tree.children[node.id])
                c_state, h_state = self.cell(node_encoding, hidden)
                node.state = (c_state.view(1, -1), h_state.view(1, -1))
            outputs.append(tree.root.state)
        return outputs


class ChildSumTreeLSTMMeanEncoder(ChildSumTreeLSTMEncoder):

    def __init__(self, input_size, hidden_size,
                 p_keep_input, p_keep_rnn):
        super(ChildSumTreeLSTMMeanEncoder, self).__init__(input_size, hidden_size,
                                                          p_keep_input, p_keep_rnn)
        self.cell = cell.BatchChildSumTreeLSTMCell(
            input_size=input_size,
            hidden_size=hidden_size,
            p_dropout=1.0 - p_keep_rnn)

    def forward(self, forest):
        outputs = []
        for tree in forest:
            for node in tree:
                node_encoding = self._get_input(node.encoding)
                hidden = self._get_hidden(tree.children[node.id])
                c_state, h_state = self.cell(node_encoding, hidden)
                node.state = (c_state.view(1, -1), h_state.view(1, -1))
            outputs.append(tree.root.state)
        return outputs
