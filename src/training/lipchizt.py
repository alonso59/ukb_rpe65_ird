import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.utils.parametrize as parametrize

@torch.enable_grad()
def _norm_gradient_sq(linear_fn, v):
    v = Variable(v, requires_grad=True)
    loss = torch.norm(linear_fn(v))**2
    loss.backward()
    return v.grad.data

def spectral_norm_approx(w,  input_size, ks, padding, stride, eps, max_iter):
    conv = torch.nn.Conv3d(w.shape[0], w.shape[1], kernel_size=ks, bias=False, padding=padding, stride=stride)
    conv.weight.data = w.data
    input_shape = (1, w.shape[1], input_size, input_size, input_size)

    v = torch.randn(input_shape, device=w.device)
    v = F.normalize(v.view(v.shape[0], -1), p=2, dim=1).view(input_shape)

    stop_criterion = False
    it = 0
    while not stop_criterion:
        previous = v
        v = _norm_gradient_sq(conv, v)
        v = F.normalize(v.view(v.shape[0], -1), p=2, dim=1).view(input_shape)
        stop_criterion = (torch.norm(v - previous) < eps) or (it > max_iter)
        it += 1
    u = conv(Variable(v))  # unormalized left singular vector
    singular_value = torch.norm(u).item()
    return singular_value

class L2Lipschitz(nn.Module):
    def _init_(self, in_size, ks, padding, stride, eps=1e-8, iterations=1, max_lc=1.0):
        super(L2Lipschitz, self)._init_()
        self.in_size = in_size
        self.padding = padding
        self.stride = stride
        self.ks = ks
        self.eps = eps
        self.iterations = iterations
        self.max_lc = max_lc

    def forward(self, x):
        norm = spectral_norm_approx(x, self.in_size, self.ks, self.padding, self.stride, self.eps, self.iterations)
        # return x / (max(1.0, (1.0 / self.max_lc) * norm))
        return x * (1.0 / max(1.0, norm / max_lc))

class L2LipschitzConv3d(nn.Module):
    def _init_(self, in_size, ks, padding, stride, eps=1e-8, iterations=1, max_lc=1.0):
        super(L2LipschitzConv3d, self)._init_()
        self.in_size = in_size
        self.padding = padding
        self.stride = stride
        self.ks = ks
        self.eps = eps
        self.iterations = iterations
        self.max_lc = max_lc

    def forward(self, w):
        norm = self.spectral_norm(w, self.in_size, self.ks, self.padding, self.stride, self.eps, self.iterations)
        # return x / (max(1.0, (1.0 / self.max_lc) * norm))
        return w * (1.0 / max(1.0, norm / max_lc))

    def spectral_norm(self, w, input_size, ks, padding, stride, eps, max_iter):
        conv = torch.nn.Conv3d(w.shape[0], w.shape[1], kernel_size=ks, bias=False, padding=padding, stride=stride)
        conv.weight.data = w.data
        input_shape = (1, w.shape[1], input_size, input_size, input_size)

        v = torch.randn(input_shape, device=w.device)
        v = F.normalize(v.view(v.shape[0], -1), p=2, dim=1).view(input_shape)

        stop_criterion = False
        it = 0
        while not stop_criterion:
            previous = v
            v = _norm_gradient_sq(conv, v)
            v = F.normalize(v.view(v.shape[0], -1), p=2, dim=1).view(input_shape)
            stop_criterion = (torch.norm(v - previous) < eps) or (it > max_iter)
            it += 1
        u = conv(Variable(v))  # unormalized left singular vector
        singular_value = torch.norm(u).item()
        return singular_value


class L2LipschitzConvTranspose3d(nn.Module):
    def _init_(self, in_size, ks, padding, stride, eps=1e-8, iterations=1, max_lc=1.0):
        super(L2LipschitzConvTranspose3d, self)._init_()
        self.in_size = in_size
        self.padding = padding
        self.stride = stride
        self.ks = ks
        self.eps = eps
        self.iterations = iterations
        self.max_lc = max_lc

    def forward(self, w):
        norm = self.spectral_norm(w, self.in_size, self.ks, self.padding, self.stride, self.eps, self.iterations)
        # return x / (max(1.0, (1.0 / self.max_lc) * norm))
        return w * (1.0 / max(1.0, norm / max_lc))

    def spectral_norm(self, w, input_size, ks, padding, stride, eps, max_iter):
        tc = nn.ConvTranspose3d(w.shape[0], w.shape[1], kernel_size=ks, stride=stride, padding=padding, bias=False)
        tc.weight.data = w.data
        input_shape = (1, w.shape[0], input_size, input_size, input_size)

        v = torch.randn(input_shape, device=w.device)
        v = F.normalize(v.view(v.shape[0], -1), p=2, dim=1).view(input_shape)

        stop_criterion = False
        it = 0
        while not stop_criterion:
            previous = v
            v = _norm_gradient_sq(tc, v)
            v = F.normalize(v.view(v.shape[0], -1), p=2, dim=1).view(input_shape)
            stop_criterion = (torch.norm(v - previous) < eps) or (it > max_iter)
            it += 1
        u = tc(Variable(v))  # unormalized left singular vector
        singular_value = torch.norm(u).item()
        return singular_value


class NetC(nn.Module):
    def _init_(self, in_ch, out_ch, ks, padding, stride, in_size, max_lc=1.):
        super(NetC, self)._init_()
        self.conv = nn.Conv3d(in_ch, out_ch, kernel_size=ks, padding=padding, stride=stride)
        parametrize.register_parametrization(self.conv, 'weight', L2LipschitzConv3d(in_size, ks, padding, stride, max_lc))

    def forward(self, x):
        return F.elu(self.conv(x))


class NetTC(nn.Module):
    def _init_(self, in_ch, out_ch, ks, padding, stride, in_size, max_lc):
        super(NetTC, self)._init_()
        self.upsample = nn.ConvTranspose3d(in_ch, out_ch, ks, stride, padding)
        parametrize.register_parametrization(self.upsample, 'weight', L2LipschitzConvTranspose3d(in_size, ks, padding, stride, max_lc))

    def forward(self, x):
        return F.elu(self.upsample(x))


def lipschitz_bn(bn_layer,x):
    return max(abs(bn_layer.weight / torch.sqrt(x)))
    
def lipschitz_bn(bn_layer):
    return max(abs(bn_layer.weight / torch.sqrt(bn_layer.running_var + bn_layer.eps)))

class Net(nn.Module):
    def _init_(self, in_ch, out_ch, ks, padding, stride, in_size, max_lc=1.):
        super(Net, self)._init_()
        self.conv = nn.Conv3d(in_ch, out_ch, kernel_size=ks, padding=padding, stride=stride)

        parametrize.register_parametrization(self.conv, 'weight', L2Lipschitz(in_size, ks, padding, stride, max_lc))
        # parametrize.register_parametrization(self.conv, 'weight', L2Lipschitz(max_k=lipschitz_bound_per_layer,
        #                                                                       in_shape=(5, 1, 16, 16, 16),
        #                                                                       stride=1,
        #                                                                       padding=1, iterations=1))

    def forward(self, x):
        return F.elu(self.conv(x))

if __name__ == '_main_':
    input = torch.rand((5, 1, 16, 16, 16))
    target = torch.rand((5, 1, 8, 8, 8))

    max_lc = 0.01
    ks = 3
    padding = 1
    stride = 2
    conv = nn.Sequential(Net(in_ch=1, out_ch=1, ks=ks, padding=padding, stride=stride, in_size=16, max_lc=max_lc))
    optimizer = optim.Adam(conv.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()

    for i in range(10):
        result = conv(input)
        loss = loss_fn(result, target)
    
        lc = (torch.tensor(spectral_norm_approx(list(list(conv.children())[0].children())[0].weight, 16, ks, padding, stride, 1e-8, 3)))
        print('epoch: ', i, '  loss: ', loss.item(), '  lipschitz_constant: ', lc.item())
        loss.backward()
        optimizer.step()