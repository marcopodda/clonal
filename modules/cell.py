import torch
import torch.nn as nn
import torch.nn.functional as F


class BatchChildSumTreeLSTMCell(nn.Module):

    def __init__(self, input_size, hidden_size, p_dropout):
        super(BatchChildSumTreeLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(p=p_dropout)

        self.ix = nn.Linear(input_size, hidden_size)
        self.ih = nn.Linear(hidden_size, hidden_size)

        self.fh = nn.Linear(hidden_size, hidden_size)
        self.fx = nn.Linear(input_size, hidden_size)

        self.ux = nn.Linear(input_size, hidden_size)
        self.uh = nn.Linear(hidden_size, hidden_size)

        self.ox = nn.Linear(input_size, hidden_size)
        self.oh = nn.Linear(hidden_size, hidden_size)

    def forward(self, inputs, previous_states):
        child_c, child_h = previous_states
        child_h_sum = F.torch.sum(torch.squeeze(child_h, 1), 0, keepdim=True)

        i = F.sigmoid(self.ix(inputs)+self.ih(child_h_sum))
        o = F.sigmoid(self.ox(inputs)+self.oh(child_h_sum))
        u = F.tanh(self.ux(inputs)+self.uh(child_h_sum))
        u = self.dropout(u)

        # add extra singleton dimension
        fx = F.torch.unsqueeze(self.fx(inputs), 1)
        f = F.torch.cat([self.fh(child_hi) + torch.squeeze(fx, 1)
                         for child_hi in child_h], 0)
        # f = torch.squeeze(f, 0)
        f = F.sigmoid(f)
        # removing extra singleton dimension
        f = F.torch.unsqueeze(f, 1)
        fc = F.torch.squeeze(F.torch.mul(f, child_c), 1)

        c = F.torch.mul(i, u) + F.torch.sum(fc, 0, keepdim=True)
        h = F.torch.mul(o, F.tanh(c))

        return c, h
