import torch
from torch import nn
from torch import optim
from torch import autograd
from torch.autograd import Variable
from torch.nn import functional as F


class Naive(nn.Module):
    """
    Define your network here.
    """

    def __init__(self, input_size, args):
        super(Naive, self).__init__()
        self.detector = nn.Sequential(
            nn.Linear(input_size, 100),
            nn.ReLU(),
            nn.Linear(100, 300),
            nn.ReLU(),
            nn.Linear(300, 300),
            nn.ReLU(),
            nn.Linear(300, 100),
            nn.ReLU(),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 2 * args.L),
        )
        self.criterion1 = nn.MSELoss()
        self.criterion2 = nn.NLLLoss()
        self.cuda = not args.no_cuda and torch.cuda.is_available()
        self.loss_alpha = args.loss_alpha
        self.temp = torch.nn.parameter.Parameter(
            data=torch.zeros(args.L, 2), requires_grad=True
        )
        self.L = args.L

        print(self)

    def forward(self, y, y_vec, h, pilot):
        device = torch.device("cuda" if self.cuda else "cpu")
        pilot_length = pilot.shape[0]
        h_hat = h[0]
        pilot = torch.view_as_real(pilot[:, -1])
        pilot_real_class = torch.where(pilot[:, 0] == -1.0, 0, 1).type(torch.long)

        y_vec = torch.flatten(torch.view_as_real(y_vec), -2, -1)
        x_prob = self.detector(y_vec)

        h_hat_mag = h_hat.abs()
        weighted_prob_list = []
        s_hat_list = []
        for l in range(self.L):
            start = self.L - 1 - l
            sharpen_prob_tmp = F.softmax(
                torch.exp(self.temp[l]) * x_prob[:, 2 * l : 2 * (l + 1)], dim=-1
            )
            weighted_prob_tmp = (h_hat_mag[l] / h_hat_mag.sum()) * sharpen_prob_tmp
            s_hat_tmp = (
                weighted_prob_tmp * (torch.tensor([-1.0, 1.0]).to(device))
            ).sum(-1)
            weighted_prob_tmp[:start] = 0
            s_hat_tmp[:start] = 0
            weighted_prob_list.append(weighted_prob_tmp.unsqueeze(-1))
            s_hat_list.append(s_hat_tmp.unsqueeze(-1))
        weighted_prob = torch.cat(weighted_prob_list, -1).sum(-1)
        s_hat = torch.cat(s_hat_list, -1)

        y_hat = torch.view_as_real((s_hat * h_hat).sum(-1))
        loss1 = self.loss_alpha * self.criterion1(
            y_hat.type(torch.FloatTensor).to(device),
            torch.view_as_real(y).type(torch.FloatTensor).to(device),
        )
        loss2 = self.criterion2(
            torch.log(weighted_prob[(self.L - 1) : (self.L - 1 + pilot_length)]),
            pilot_real_class,
        )

        loss = loss1 + loss2

        return loss, y_hat, x_prob, weighted_prob, loss1, loss2
