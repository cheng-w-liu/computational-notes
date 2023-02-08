from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F


def forward_pass(net, x, y, weights=None):
    pred = net.net_forward(x, weights)
    loss = net.loss_func(pred, y)
    return loss, pred


class SinusoidalNet(nn.Module):

    def __init__(self, input_dim, hidden_dim, loss_func, p_drop=0.01):
        super(SinusoidalNet, self).__init__()
        self.net = nn.Sequential(
            OrderedDict(
                [
                    ("linear1", nn.Linear(input_dim, hidden_dim)),
                    ("relu1", nn.ReLU()),
                    ("linear2", nn.Linear(hidden_dim, hidden_dim)),
                    ("relu2", nn.ReLU()),
                    ("linear3", nn.Linear(hidden_dim, 1)),
                ]
            )
        )
        self.loss_func = loss_func

    def forward(self, x, weights=None):
        if weights is None:
            x = self.net(x)
            x = x.squeeze(1)
        else:
            x = F.linear(x, weights["net.linear1.weight"], weights["net.linear1.bias"])
            x = F.relu(x)

            x = F.linear(x, weights["net.linear2.weight"], weights["net.linear2.bias"])
            x = F.relu(x)

            x = F.linear(x, weights["net.linear3.weight"], weights["net.linear3.bias"])
            x = x.squeeze(1)
        return x

    def net_forward(self, x, weights=None):
        return self.forward(x, weights)

    def copy_weights(self, net):
        for m_from, m_to in zip(net.modules(), self.modules()):
            if isinstance(m_to, nn.Linear) or isinstance(m_to, nn.Conv2d) or isinstance(m_to, nn.BatchNorm1d) or isinstance(m_to, nn.BatchNorm2d):
                m_to.weight.data = m_from.weight.data.clone()
                if m_to.bias is not None:
                    m_to.bias.data = m_from.bias.data.clone()


class InnerLoop(SinusoidalNet):

    def __init__(self, input_dim, hidden_dim, loss_func, num_updates, step_size, meta_batch_size, p_drop=0.01):
        super(InnerLoop, self).__init__(input_dim, hidden_dim, loss_func, p_drop)
        self.num_updates = num_updates
        self.step_size = step_size
        self.meta_batch_size = meta_batch_size

    def net_forward(self, x, weights=None):
        return super(InnerLoop, self).forward(x, weights)

    def forward_pass(self, x, y, weights=None):
        pred = self.net_forward(x, weights)
        loss = self.loss_func(pred, y)
        return loss, pred

    def forward(self, support_x, support_y, query_x, query_y):

        fast_weights = OrderedDict(
            (name, param) for (name, param) in self.named_parameters()
        )

        for i in range(self.num_updates):
            if i == 0:
                loss, _ = self.forward_pass(support_x, support_y)
                grads = torch.autograd.grad(loss, self.parameters(), create_graph=True)
            else:
                loss, _ = self.forward_pass(support_x, support_y, fast_weights)
                grads = torch.autograd.grad(loss, fast_weights.values(), create_graph=True)

            fast_weights = OrderedDict(
                (name, param - self.step_size * g)
                for ((name, param), g) in zip(fast_weights.items(), grads)
            )

        loss, _ = self.forward_pass(query_x, query_y, fast_weights)
        query_loss = loss.data.detach().numpy()
        loss = loss / self.meta_batch_size
        grads = torch.autograd.grad(loss, self.parameters())
        meta_grads = {name: g for ((name, _), g) in zip(self.named_parameters(), grads)}

        return query_loss, meta_grads


class MetaLearner(object):
    def __init__(self, meta_batch_size, meta_step_size, inner_step_size, num_updates, num_inner_updates, loss_func, input_dim, hidden_dim, p_drop=0.01):
        super(self.__class__, self).__init__()
        self.meta_batch_size = meta_batch_size
        self.meta_step_size = meta_step_size
        self.inner_step_size = inner_step_size
        self.num_updates = num_updates
        self.num_inner_updates = num_inner_updates
        self.loss_func = loss_func
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.p_drop = p_drop
        self.net = SinusoidalNet(input_dim, hidden_dim, loss_func, p_drop)
        self.fast_net = InnerLoop(input_dim, hidden_dim, loss_func, num_inner_updates, inner_step_size, meta_batch_size, p_drop)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=meta_step_size)

    def fine_tune_and_predict(self, support_x, support_y, query_x, fine_tune_steps=None):
        if fine_tune_steps is None:
            fine_tune_steps = self.num_inner_updates

        test_net = SinusoidalNet(self.input_dim, self.hidden_dim, self.loss_func, self.p_drop)
        test_net.copy_weights(self.net)
        test_opt = torch.optim.SGD(test_net.parameters(), lr=self.inner_step_size)
        for i in range(fine_tune_steps):
            loss, pred = forward_pass(test_net, support_x, support_y)
            test_opt.zero_grad()
            loss.backward()
            test_opt.step()

        pred = test_net.net_forward(query_x)
        return pred

    def meta_update(self, task_grads, dummy_x, dummy_y):
        loss, pred = forward_pass(self.net, dummy_x, dummy_y)
        gradients = {
            k: sum(grads[k] for grads in task_grads) for k in task_grads[0].keys()
        }
        hooks = []
        for k, v in self.net.named_parameters():

            def get_closure():
                key = k
                def replace_grad(arg):
                    return gradients[key]
                return replace_grad

            hooks.append(v.register_hook(get_closure()))

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        for h in hooks:
            h.remove()

    def train(self, support_x_list, support_y_list, query_x_list, query_y_list):
        assert self.meta_batch_size == len(support_x_list)

        query_loss = None

        for i in range(self.num_updates):
            task_grads = []

            total_query_loss = 0.
            total_query_size = 0.

            for b in range(self.meta_batch_size):
                support_x = support_x_list[b]
                support_y = support_y_list[b]
                query_x = query_x_list[b]
                query_y = query_y_list[b]

                self.fast_net.copy_weights(self.net)
                loss, grads = self.fast_net.forward(support_x, support_y, query_x, query_y)

                task_grads.append(grads)
                total_query_loss += loss * len(query_y)
                total_query_size += len(query_y)

            self.meta_update(task_grads, support_x_list[0], support_y_list[0])

            if i == self.num_updates - 1:
                query_loss = total_query_loss / total_query_size

        return np.sqrt(query_loss)

    def save_model(self, save_path):
        custom_state_dict = {
            'net': self.net.state_dict(),
            'optimizer': self.opt.state_dict(),
        }
        torch.save(custom_state_dict, save_path)

    def load_model(self, load_path):
        checkpoint = torch.load(load_path)
        self.net.load_state_dict(checkpoint['net'])
        self.opt.load_state_dict(checkpoint['optimizer'])











