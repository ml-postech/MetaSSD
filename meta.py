import torch
from torch import nn
from torch import optim
from torch import autograd

from torch.nn import functional as F
import numpy as np


class Learner(nn.Module):
    """
    This is a learner class, which will accept a specific network module, such as OmniNet that define the network forward
    process. Learner class will create two same network, one as theta network and the other acts as theta_pi network.
    for each episode, the theta_pi network will copy its initial parameters from theta network and update several steps
    by meta-train set and then calculate its loss on meta-test set. All loss on meta-test set will be sumed together and
    then backprop on theta network, which should be done on metalaerner class.
    For learner class, it will be responsible for update for several steps on meta-train set and return with the loss on
    meta-test set.
    """

    def __init__(self, update_lr, temp_lr, reg2, net_cls, *args):
        """
        It will receive a class: net_cls and its parameters: args for net_cls.
        :param net_cls: class, not instance
        :param args: the parameters for net_cls
        """
        super(Learner, self).__init__()
        # pls make sure net_cls is a class but NOT an instance of class.
        assert net_cls.__class__ == type

        # we will create two class instance meanwhile and use one as theta network and the other as theta_pi network.
        self.net = net_cls(*args)
        # you must call create_pi_net to create pi network additionally
        self.net_pi = net_cls(*args)
        # update theta_pi = theta_pi - lr * grad
        # according to the paper, here we use naive version of SGD to update theta_pi
        # 0.1 here means the learner_lr
        temp_param = [v for k, v in self.net_pi.named_parameters() if "temp" in k]
        encoder_param = [
            v for k, v in self.net_pi.named_parameters() if "temp" not in k
        ]
        param_list = [
            {"params": temp_param, "lr": temp_lr, "weight_decay": reg2},
            {"params": encoder_param, "lr": update_lr},
        ]
        self.optimizer = optim.SGD(param_list)

    def parameters(self):
        """
        Override this function to return only net parameters for MetaLearner's optimize
        it will ignore theta_pi network parameters.
        :return:
        """
        return self.net.parameters()

    def update_pi(self):
        """
        copy parameters from self.net -> self.net_pi
        :return:
        """
        for m_from, m_to in zip(self.net.modules(), self.net_pi.modules()):
            if (
                isinstance(m_to, nn.Linear)
                or isinstance(m_to, nn.Conv2d)
                or isinstance(m_to, nn.BatchNorm2d)
            ):
                m_to.weight.data = m_from.weight.data.clone()
                if m_to.bias is not None:
                    m_to.bias.data = m_from.bias.data.clone()

    def forward(
        self,
        y_qry,
        y_vec_qry,
        h_qry,
        x_qry,
        s_qry,
        num_updates,
        pilot_length,
    ):
        # now try to fine-tune from current $theta$ parameters -> $theta_pi$
        # after num_updates of fine-tune, we will get a good theta_pi parameters so that it will retain satisfying
        # performance on specific task, that's, current episode.
        # firstly, copy theta_pi from theta network
        self.update_pi()

        L = h_qry.shape[-1]
        y_spt = y_qry[(L - 1) : (L - 1 + pilot_length)]
        s_spt = s_qry[(L - 1) : (L - 1 + pilot_length)]

        symbol = torch.view_as_real(s_qry[:, -1])
        real_class = torch.where(symbol[:, 0] == -1.0, 0, 1).type(torch.long)

        # update for several steps
        for i in range(num_updates):
            # forward and backward to update net_pi grad.
            loss, y_hat, x_prob, weighted_prob, loss1, loss2 = self.net_pi(
                y_qry, y_vec_qry, h_qry, s_spt
            )
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net_pi.parameters(), 1.0)
            self.optimizer.step()

        # Compute the meta gradient and return it, the gradient is from one episode
        # in metalearner, it will merge all loss from different episode and sum over it.
        loss, y_hat, x_prob, weighted_prob, loss1, loss2 = self.net_pi(
            y_qry, y_vec_qry, h_qry, s_spt
        )

        accuracy = (
            (weighted_prob.argmax(1) == real_class).type(torch.float).mean().item()
        )
        ber_acc = accuracy

        # gradient for validation on theta_pi
        # after call autorad.grad, you can not call backward again except for setting create_graph = True
        # as we will use the loss as dummpy loss to conduct a dummy backprop to write our gradients to theta network,
        # here we set create_graph to true to support second time backward.
        grads_pi = autograd.grad(loss, self.net_pi.parameters(), create_graph=True)

        return loss, grads_pi, accuracy, loss1, loss2

    def net_forward(self, y_spt, y1_spt, y2_spt, s_spt):
        """
        This function is purely for updating net network. In metalearner, we need the get the loss op from net network
        to write our merged gradients into net network, hence will call this function to get a dummy loss op.
        :param support_x: [setsz, c, h, w]
        :param support_y: [sessz, c, h, w]
        :return: dummy loss and dummy pred
        """
        loss, y_hat, x_prob, weighted_prob, loss1, loss2 = self.net(
            y_spt, y1_spt, y2_spt, s_spt
        )
        return loss, y_hat, x_prob


class MetaLearner(nn.Module):
    """
    As we have mentioned in Learner class, the metalearner class will receive a series of loss on different tasks/episodes
    on theta_pi network, and it will merage all loss and then sum over it. The summed loss will be backproped on theta
    network to update theta parameters, which is the initialization point we want to find.
    """

    def __init__(
        self,
        net_cls,
        net_cls_args,
        meta_batchsz,
        update_lr,
        temp_lr,
        meta_lr,
        gamma,
        num_updates,
        pl,
        reg,
        reg2,
    ):
        """

        :param net_cls: class, not instance. the class of specific Network for learner
        :param net_cls_args: tuple, args for net_cls, like (n_way, imgsz)
        :param n_way:
        :param k_shot:
        :param meta_batchsz: number of tasks/episode
        :param meta_lr: learning rate for meta-learner
        :param num_updates: number of updates for learner
        """
        super(MetaLearner, self).__init__()

        self.meta_batchsz = meta_batchsz
        self.meta_lr = meta_lr
        self.gamma = gamma
        self.num_updates = num_updates
        self.pilot_length = pl

        # it will contains a learner class to learn on episodes and gather the loss together.
        self.learner = Learner(update_lr, temp_lr, reg2, net_cls, *net_cls_args)
        # the optimizer is to update theta parameters, not theta_pi parameters.
        temp_param = [v for k, v in self.learner.named_parameters() if "temp" in k]
        encoder_param = [
            v for k, v in self.learner.named_parameters() if "temp" not in k
        ]
        param_list = [
            {"params": temp_param, "lr": 0.0},
            {"params": encoder_param, "lr": meta_lr, "weight_decay": reg},
        ]
        self.optimizer = optim.SGD(param_list)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=gamma)

    def write_grads(self, dummy_loss, sum_grads_pi):
        """
        write loss into learner.net, gradients come from sum_grads_pi.
        Since the gradients info is not calculated by general backward, we need this function to write the right gradients
        into theta network and update theta parameters as wished.
        :param dummy_loss: dummy loss, nothing but to write our gradients by hook
        :param sum_grads_pi: the summed gradients
        :return:
        """

        # Register a hook on each parameter in the net that replaces the current dummy grad
        # with our grads accumulated across the meta-batch
        hooks = []

        for i, v in enumerate(self.learner.parameters()):

            def closure():
                ii = i
                return lambda grad: sum_grads_pi[ii]

            # if you write: hooks.append( v.register_hook(lambda grad : sum_grads_pi[i]) )
            # it will pop an ERROR, i don't know why?
            hooks.append(v.register_hook(closure()))

        # use our sumed gradients_pi to update the theta/net network,
        # since our optimizer receive the self.net.parameters() only.
        self.optimizer.zero_grad()
        dummy_loss.backward()
        self.optimizer.step()

        # if you do NOT remove the hook, the GPU memory will expode!!!
        for h in hooks:
            h.remove()

    def forward(self, y_qry, y_vec_qry, h_qry, x_qry, s_qry):
        sum_grads_pi = None
        meta_batchsz = y_qry.size(0)

        # we do different learning task sequentially, not parallel.
        accs = []
        losses = []
        loss1s = []
        loss2s = []
        # for each task/episode.
        for i in range(meta_batchsz):
            batch_loss, grad_pi, episode_acc, loss1, loss2 = self.learner(
                y_qry[i],
                y_vec_qry[i],
                h_qry[i],
                x_qry[i],
                s_qry[i],
                self.num_updates,
                self.pilot_length,
            )
            accs.append(episode_acc)
            losses.append(batch_loss.detach().cpu().item())
            loss1s.append(loss1.detach().cpu().item())
            loss2s.append(loss2.detach().cpu().item())
            if sum_grads_pi is None:
                sum_grads_pi = grad_pi
            else:  # accumulate all gradients from different episode learner
                sum_grads_pi = [torch.add(i, j) for i, j in zip(sum_grads_pi, grad_pi)]

        # As we already have the grads to update
        # We use a dummy forward / backward pass to get the correct grads into self.net
        # the right grads will be updated by hook, ignoring backward.
        # use hook mechnism to write sumed gradient into network.
        # we need to update the theta/net network, we need a op from net network, so we call self.learner.net_forward
        # to get the op from net network, since the loss from self.learner.forward will return loss from net_pi network.
        # dummy_loss, _ = self.learner.net_forward(support_x[0], support_y[0])
        L = h_qry.shape[-1]
        dummy_loss, _, _ = self.learner.net_forward(
            y_qry[0],
            y_vec_qry[0],
            h_qry[0],
            s_qry[0, (L - 1) : (L - 1 + self.pilot_length)],
        )
        self.write_grads(dummy_loss, sum_grads_pi)

        return accs, losses, loss1s, loss2s, self.scheduler

    def pred(self, y_qry, y_vec_qry, h_qry, x_qry, s_qry):
        """
        predict for query_x
        :param support_x:
        :param support_y:
        :param query_x:
        :param query_y:
        :return:
        """
        meta_batchsz = y_qry.size(0)

        accs = []
        losses = []
        loss1s = []
        loss2s = []
        # for each task/episode.
        # the learner will copy parameters from current theta network and then fine-tune on support set.
        for i in range(meta_batchsz):
            loss, _, episode_acc, _, _ = self.learner(
                y_qry[i],
                y_vec_qry[i],
                h_qry[i],
                x_qry[i],
                s_qry[i],
                self.num_updates,
                self.pilot_length,
            )
            accs.append(episode_acc)
            losses.append(loss.detach().cpu())

        return np.array(accs), np.array(losses)
