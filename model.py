""" Pansharpening Based on Multi-Sequence Convolutional Recurrent Network """

import torch.nn as nn
import torch
from torch.autograd import Variable
from torchsummary import summary
import torch.nn as nn
import torch
from torch.autograd import Variable
from torchsummary import summary
class ResidualBlock(nn.Module):
    def __init__(self, channel_size, kernel_size):
        super(ResidualBlock, self).__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=channel_size, out_channels=channel_size, kernel_size=kernel_size, padding=padding),
            nn.PReLU(),
            nn.Conv2d(in_channels=channel_size, out_channels=channel_size, kernel_size=kernel_size, padding=padding),
            nn.PReLU()
        )
    def forward(self, x):
        residual = self.block(x)
        return x + residual


class EncoderUnit(nn.Module):
    def __init__(self, in_channels, num_res_layers, kernel_size, channel_size):
        super(EncoderUnit, self).__init__()
        padding = kernel_size // 2
        self.init_layer = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=channel_size, kernel_size=kernel_size, padding=padding),
            nn.PReLU())

        res_layers = [ResidualBlock(channel_size, kernel_size) for _ in range(num_res_layers)]
        self.res_layers = nn.Sequential(*res_layers)

        self.final = nn.Sequential(
            nn.Conv2d(in_channels=channel_size, out_channels=32, kernel_size=kernel_size, padding=padding)
        )

    def forward(self, x):
        x = self.init_layer(x)
        x = self.res_layers(x)
        x = self.final(x)
        return x


class ConvGRUUnit(nn.Module):
    def __init__(self, in_channels, hidden_channels, kernel_size):
        super(ConvGRUUnit, self).__init__()
        padding = kernel_size // 2
        self.hidden_channels = hidden_channels
        self.reset_gate = nn.Conv2d(in_channels + hidden_channels, hidden_channels, kernel_size, padding=padding)
        self.update_gate = nn.Conv2d(in_channels + hidden_channels, hidden_channels, kernel_size, padding=padding)
        self.out_gate = nn.Conv2d(in_channels + hidden_channels, hidden_channels, kernel_size, padding=padding)

    def forward(self, x, prev_state):
        if prev_state is None:
            state_size = [x.shape[0], self.hidden_channels] + list(x.shape[2:])
            prev_state = Variable(torch.zeros(state_size))
            if torch.cuda.is_available():
                prev_state = prev_state.cuda()

        stacked = torch.cat([x, prev_state], dim=1)

        update = torch.sigmoid(self.update_gate(stacked))
        reset = torch.sigmoid(self.reset_gate(stacked))
        candidate_state = torch.tanh(self.out_gate(torch.cat([x, prev_state * reset], dim=1)))
        new_state = (1 - update) * prev_state + update * candidate_state

        return new_state


class FusionModule(nn.Module):
    def __init__(self):
        '''
        Args:
            fuse_config : dict, fusion configuration
        '''
        super(FusionModule, self).__init__()
        self.input_channels =1
        self.num_hidden_layers =2
        hidden_channels = [32,32]
        kernel_sizes =[3,3]

        gru_units = []
        for i in range(0, self.num_hidden_layers):
            cur_input_dim = hidden_channels[i - 1]
            gru_unit = ConvGRUUnit(in_channels=cur_input_dim, hidden_channels=hidden_channels[i],
                                   kernel_size=kernel_sizes[i])
            gru_units.append(gru_unit)
        self.gru_units = nn.ModuleList(gru_units)
    def forward(self, x, alphas, h=None):
        if h is None:
            hidden_states = [None] * self.num_hidden_layers
        num_low_res = x.shape[1]
        cur_layer_input = x
        for l in range(self.num_hidden_layers):
            gru_unit = self.gru_units[l]
            h = hidden_states[l]

            out = []
            for t in range(num_low_res):
                h = gru_unit(cur_layer_input[:, t, :, :, :], h)
                out.append(h)
            out = torch.stack(out, dim=1)

            cur_layer_input = out
        fused_representations = torch.sum(cur_layer_input * alphas, 1) / torch.sum(alphas, 1)

        return fused_representations


class Decoder(nn.Module):
    def __init__(self):
        '''
        Args:
            dec_config : dict, decoder configuration
        '''

        super(Decoder, self).__init__()

        self.final = nn.Conv2d(in_channels=32,
                               out_channels=4,
                               kernel_size=1,
                               padding=1// 2)

    def forward(self, x):

        x = self.final(x)

        return x


class fusion(nn.Module):
    def __init__(self):
        '''
        Args:
            config : dict, configuration file
        '''

        super(fusion, self).__init__()
        self.unit1 = EncoderUnit(in_channels=1,
                                 num_res_layers=2,
                                 kernel_size=3,
                                 channel_size=32)
        self.fuse = FusionModule()
        self.decode = Decoder()

    def forward(self, lrs):
        batch_size, num_low_res, height, width = lrs.shape
        lrs = lrs.view(batch_size, num_low_res, 1, height, width)
        alphas = torch.ones(2, 9)
        alphas = alphas.float().cuda()
        alphas = alphas.view(-1, num_low_res, 1, 1, 1)
        lrs = lrs.view(batch_size * num_low_res, 1, height, width)
        lrs = self.unit1(lrs)
        out = lrs.view(batch_size, num_low_res, -1, height, width)
        out = self.fuse(out, alphas)
        out = self.decode(out)

        return out
