import torch
import torch.nn as nn
import numpy as np
from . import modules


class CondGlowStep(nn.Module):

    def __init__(self, x_size, y_size, x_hidden_channels, x_hidden_size, y_hidden_channels):


        super().__init__()

        # 1. cond-actnorm
        self.actnorm = modules.CondActNorm(x_size=x_size, y_channels=y_size[0], x_hidden_channels=x_hidden_channels, x_hidden_size=x_hidden_size)

        # 2. cond-1x1conv
        self.invconv = modules.Cond1x1Conv(x_size=x_size, x_hidden_channels=x_hidden_channels, x_hidden_size=x_hidden_size, y_channels=y_size[0])

        # 3. cond-affine
        self.affine = modules.CondAffineCoupling(x_size=x_size, y_size=[y_size[0] // 2, y_size[1], y_size[2]], hidden_channels=y_hidden_channels)


    def forward(self, x, y, logdet=None, reverse=False):

        if reverse is False:
            # 1. cond-actnorm
            y, logdet = self.actnorm(x, y, logdet, reverse=False)

            # 2. cond-1x1conv
            y, logdet = self.invconv(x, y, logdet, reverse=False)

            # 3. cond-affine
            y, logdet = self.affine(x, y, logdet, reverse=False)

            # Return
            return y, logdet


        if reverse is True:
            # 3. cond-affine
            y, logdet = self.affine(x, y, logdet, reverse=True)

            # 2. cond-1x1conv
            y, logdet = self.invconv(x, y, logdet, reverse=True)

            # 1. cond-actnorm
            y, logdet = self.actnorm(x, y, logdet, reverse=True)

            # Return
            return y, logdet


class CondGlow(nn.Module):

    def __init__(self, x_size, y_size, x_hidden_channels, x_hidden_size, y_hidden_channels, K, L):


        super().__init__()
        self.layers = nn.ModuleList()
        self.output_shapes = []
        self.K = K
        self.L = L
        C, H, W = y_size

        for l in range(0, L):

            # 1. Squeeze
            C, H, W = C * 4, H // 2, W // 2
            y_size = [C,H,W]
            self.layers.append(modules.SqueezeLayer(factor=2))
            self.output_shapes.append([-1, C, H, W])

            # 2. K CGlowStep
            for k in range(0, K):

                self.layers.append(CondGlowStep(x_size = x_size,
                                            y_size = y_size,
                                            x_hidden_channels = x_hidden_channels,
                                            x_hidden_size = x_hidden_size,
                                            y_hidden_channels = y_hidden_channels,
                                            )
                                   )

                self.output_shapes.append([-1, C, H, W])

            # 3. Split
            if l < L - 1:
                self.layers.append(modules.Split2d(num_channels=C))
                self.output_shapes.append([-1, C // 2, H, W])
                C = C // 2


    def forward(self, x, y, logdet=0.0, reverse=False, eps_std=1.0):
        if reverse == False:
            return self.encode(x, y, logdet)
        else:
            return self.decode(x, y, logdet, eps_std)

    def encode(self, x, y, logdet=0.0):
        for layer, shape in zip(self.layers, self.output_shapes):
            if isinstance(layer, modules.Split2d) or isinstance(layer, modules.SqueezeLayer):
                y, logdet = layer(y, logdet, reverse=False)

            else:
                y, logdet = layer(x, y, logdet, reverse=False)
        return y, logdet

    def decode(self, x, y, logdet=0.0, eps_std=1.0):
        for layer in reversed(self.layers):
            if isinstance(layer, modules.Split2d):
                y, logdet = layer(y, logdet=logdet, reverse=True, eps_std=eps_std)

            elif isinstance(layer, modules.SqueezeLayer):
                y, logdet = layer(y, logdet=logdet, reverse=True)

            else:
                y, logdet = layer(x, y, logdet=logdet, reverse=True)

        return y, logdet


class CondGlowModel(nn.Module):
    BCE = nn.BCEWithLogitsLoss()
    CE = nn.CrossEntropyLoss()

    def __init__(self, args):
        super().__init__()
        self.flow = CondGlow(x_size=args.x_size,
                            y_size=args.y_size,
                            x_hidden_channels=args.x_hidden_channels,
                            x_hidden_size=args.x_hidden_size,
                            y_hidden_channels=args.y_hidden_channels,
                            K=args.flow_depth,
                            L=args.num_levels,
                            )

        self.learn_top = args.learn_top


        self.register_parameter("new_mean",
                                nn.Parameter(torch.zeros(
                                    [1,
                                     self.flow.output_shapes[-1][1],
                                     self.flow.output_shapes[-1][2],
                                     self.flow.output_shapes[-1][3]])))


        self.register_parameter("new_logs",
                                nn.Parameter(torch.zeros(
                                    [1,
                                     self.flow.output_shapes[-1][1],
                                     self.flow.output_shapes[-1][2],
                                     self.flow.output_shapes[-1][3]])))

        self.n_bins = args.y_bins


    def prior(self):    # 对于有类别的样本，可以在此处指定类别附近的采样

        if self.learn_top:
            return self.new_mean, self.new_logs
        else:
            return torch.zeros_like(self.new_mean), torch.zeros_like(self.new_mean)


    def forward(self, x=0.0, y=None, eps_std=1.0, reverse=False):
        if reverse == False:
            dimensions = y.size(1)*y.size(2)*y.size(3)
            logdet = torch.zeros_like(y[:, 0, 0, 0])
            logdet += float(-np.log(self.n_bins) * dimensions)
            z, objective = self.flow(x, y, logdet=logdet, reverse=False)
            mean, logs = self.prior()
            objective += modules.GaussianDiag.logp(mean, logs, z)
            nll = -objective / float(np.log(2.) * dimensions)
            return z, nll

        else:   # 注意，根据原论文，y采样并非z直接采样后高斯映射，而是多点采样以求argmax_y p(y|x)
            with torch.no_grad():
                mean, logs = self.prior()
                if y is None:
                    for i in range(10):
                        z_i = modules.GaussianDiag.batchsample(x.size(0), mean, logs, eps_std)
                        y_i, logdet = self.flow(x, z_i, eps_std=eps_std, reverse=True)
                        y = y_i if i==0 else y+y_i
                    y = y/10
                else:
                    y, logdet = self.flow(x, y, eps_std=eps_std, reverse=True)
            return y, logdet

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='train c-Glow')

    # C-Glow parameters
    parser.add_argument("--x_size", type=tuple, default=(3,64,64))
    parser.add_argument("--y_size", type=tuple, default=(3,64,64))
    parser.add_argument("--x_hidden_channels", type=int, default=128)
    parser.add_argument("--x_hidden_size", type=int, default=64)
    parser.add_argument("--y_hidden_channels", type=int, default=256)
    parser.add_argument("-K", "--flow_depth", type=int, default=4)
    parser.add_argument("-L", "--num_levels", type=int, default=2)
    parser.add_argument("--learn_top", type=bool, default=False)
    # Dataset preprocess parameters
    parser.add_argument("--label_scale", type=float, default=1)
    parser.add_argument("--label_bias", type=float, default=0.5)
    parser.add_argument("--x_bins", type=float, default=256.0)
    parser.add_argument("--y_bins", type=float, default=256.0)
    args = parser.parse_args()
    model = CondGlowModel(args)

    input_x = torch.randn((1,3,64,64))
    input_y = torch.randn((1,3,64,64))
    latent_z,nll = model(input_x.clone(),input_y,reverse=False)   # Forward
    recover_y,nll_new = model(input_x.clone(),latent_z,reverse=True)    # Reverse
    # recover_z,nll_new = model(recover_y.clone(),input_y,reverse=False)
    print('Difference between yi and yo:',torch.sum(torch.abs(recover_y-input_y)))
    print('NLL of yi and yo:',nll,nll_new)