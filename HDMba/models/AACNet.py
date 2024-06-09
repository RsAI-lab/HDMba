import torch.nn as nn
import torch.nn.init as init
import torch
import torch.nn.functional as F
from torchsummary import summary


class FAConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, padding=0, deploy=False, reduce_gamma=False, gamma_init=None ):
        super(FAConv, self).__init__()
        self.deploy = deploy
        if deploy:
            self.fused_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(kernel_size,kernel_size), stride=stride,
                                      padding=padding, bias=False, padding_mode= padding_mode)
            # self.fused_point_conv = nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=1,stride=1,padding=0)
        else:
            self.square_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(kernel_size, kernel_size),stride=stride,padding=padding, dilation=dilation,bias=True)
            self.square_point_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1,stride=stride,bias=True)
            if padding - kernel_size // 2 >= 0:
                #   Common use case. E.g., k=3, p=1 or k=5, p=2
                self.crop = 0
                #   Compared to the KxK layer, the padding of the 1xK layer and Kx1 layer should be adjust to align the sliding windows (Fig 2 in the paper)
                hor_padding = [padding - kernel_size // 2, padding]
                ver_padding = [padding, padding - kernel_size // 2]
            else:
                #   A negative "padding" (padding - kernel_size//2 < 0, which is not a common use case) is cropping.
                #   Since nn.Conv2d does not support negative padding, we implement it manually
                self.crop = kernel_size // 2 - padding
                hor_padding = [0, padding]
                ver_padding = [padding, 0]

            self.ver_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(kernel_size, 1),
                                      stride=stride,padding=ver_padding, dilation=dilation, bias=True)
            self.hor_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, kernel_size),
                                      stride=stride,padding=hor_padding, dilation=dilation, bias=True)
            if reduce_gamma:
                self.init_gamma(1.0 / 3)
            if gamma_init is not None:
                assert not reduce_gamma
                self.init_gamma(gamma_init)

    def forward(self, input):
        if self.deploy:
            return self.fused_conv(input)
        else:
            square_outputs_dw=self.square_conv(input)
            square_outputs_pw = self.square_point_conv(input)
            square_outputs = square_outputs_dw+square_outputs_pw
            if self.crop > 0:
                ver_input = input[:, :, :, self.crop:-self.crop]
                hor_input = input[:, :, self.crop:-self.crop, :]
            else:
                ver_input = input
                hor_input = input
            vertical_outputs_dw = self.ver_conv(ver_input)
            vertical_outputs=vertical_outputs_dw
            horizontal_outputs_dw = self.hor_conv(hor_input)
            horizontal_outputs=horizontal_outputs_dw
            result = square_outputs + vertical_outputs + horizontal_outputs
            return result


class GA(nn.Module):
    def __init__(self, dim=150):
        super(GA, self).__init__()
        self.ga = nn.Sequential(*[
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1,
                      padding=0, groups=1, bias=True,padding_mode="zeros"),
            nn.ReLU(),
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1,
                      padding=0, groups=1, bias=True,padding_mode="zeros"),
            nn.Sigmoid()
        ])

    def forward(self, x):
        out=self.ga(x)
        return out


class AAConv(nn.Module):
    def __init__(self, dim,kernel_size,padding,stride,deploy):
        super(AAConv, self).__init__()
        self.faconv=FAConv(dim, dim, kernel_size=kernel_size, padding=padding, stride=stride, deploy=deploy)
        self.ga=GA(dim=dim)

    def forward(self, input):
        x=self.faconv(input)
        attn=self.ga(input)
        out=attn*x
        return out


class PCSA(nn.Module):
    def __init__(self, dim=150, out_dim=150):
        super(PCSA, self).__init__()
        self.in_dim=dim
        self.out_dim=out_dim
        self.avg_pooling=torch.nn.AdaptiveAvgPool2d((1,1))
        self.liner1=nn.Linear(dim,dim)
        self.liner2=nn.Linear(dim,dim)
        self.Sigmoid=nn.Sigmoid()
        self.conv2d = nn.Conv2d(dim,dim,1,1)
        self.avg_pooling_1d = torch.nn.AdaptiveAvgPool1d(1)
        self.conv1d=nn.Conv1d(1,1,1,1)

    def forward(self, x):
        # x=self.conv2d(x)

        x_pool=self.avg_pooling(x).squeeze(dim=3).permute(0,2,1)
        # print(x_pool.shape)
        q=self.liner1(x_pool)
        k=self.liner2(x_pool)
        attn=k.permute(0,2,1)@q
        attn = self.avg_pooling_1d(attn)
        attn = self.conv1d(attn.transpose(-1,-2)).transpose(-1,-2).unsqueeze(dim=3)
        attn=self.Sigmoid(attn)
        out =x*attn
        return out


# Pixel Attention 像素注意力
class AACNet(nn.Module):
    # def __init__(self, in_dim=4, out_dim=4, dim=64, kernel_size=3, padding=1, num_blocks=5, stride=1, deploy=False):
    # def __init__(self, in_dim=7, out_dim=7, dim=64, kernel_size=3, padding=1, num_blocks=5, stride=1, deploy=False):
    def __init__(self, in_dim=305, out_dim=305, dim=64, kernel_size=3, padding=1,num_blocks=5, stride=1,deploy=False):
        super(AACNet, self).__init__()
        self.blocks = nn.ModuleList([])
        self.blocks1 = nn.ModuleList([])
        self.blocks2 = nn.ModuleList([])
        self.dim = dim
        self.t_Conv1 = nn.Conv2d(self.dim, self.dim, 3, 1, 1)
        self.t_Conv2 = nn.Conv2d(self.dim, self.dim, 3, 1, 1)
        self.t_Conv3 = nn.Conv2d(self.dim, self.dim, 3, 1, 1)
        self.num_block = num_blocks
        self.Convd_in = nn.Conv2d(in_dim, self.dim, kernel_size=1, padding=0, stride=1)
        self.Convd = nn.Conv2d(self.dim, out_dim, kernel_size=1, padding=0, stride=1)
        self.Convd_out = nn.Conv2d(out_dim, out_dim, 3, 1, 1)
        self.cattn = PCSA(self.dim, self.dim)
        # self.acdw=AAConv( self.dim, self.dim, kernel_size=kernel_size, padding=padding, stride=stride, deploy=False)
        # self.gps=1

        def weigth_init(m):
            if isinstance(m, nn.Conv2d):
                init.xavier_uniform_(m.weight.data)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
        for _ in range(num_blocks):
            self.blocks.append(nn.ModuleList([
                AAConv(self.dim, kernel_size=kernel_size, padding=padding, stride=stride, deploy=deploy),
                nn.PReLU(),
                AAConv( self.dim, kernel_size=kernel_size, padding=padding, stride=stride, deploy=deploy),
                PCSA(self.dim,self.dim),
            ]))
        for _ in range(num_blocks):
            self.blocks1.append(nn.ModuleList([
                AAConv(self.dim, kernel_size=kernel_size, padding=padding, stride=stride, deploy=deploy),
                nn.PReLU(),
                AAConv(self.dim, kernel_size=kernel_size, padding=padding, stride=stride, deploy=deploy),
                PCSA(self.dim, self.dim),
            ]))
        for _ in range(num_blocks):
            self.blocks2.append(nn.ModuleList([
                AAConv(self.dim, kernel_size=kernel_size, padding=padding, stride=stride, deploy=deploy),
                nn.PReLU(),
                AAConv(self.dim, kernel_size=kernel_size, padding=padding, stride=stride, deploy=deploy),
                PCSA(self.dim, self.dim),
            ]))
        self.cattn.apply(weigth_init)
        self.blocks.apply(weigth_init)
        self.blocks1.apply(weigth_init)
        self.blocks2.apply(weigth_init)
        # self..apply(weigth_init)

    def forward(self, x):
        x_original_features = x
        x = self.Convd_in(x)
        x_shallow_features = x
        for (aaconv, act, aaconv1, pcsa) in self.blocks:
            res = x
            x = aaconv(x)
            x = act(x)
            x = aaconv1(x)
            x = pcsa(x)
            x = x+res
        x = self.t_Conv1(x)
        for (aaconv, act, aaconv1, pcsa) in self.blocks1:
            res1 = x
            x = aaconv(x)
            x = act(x)
            x = aaconv1(x)
            x = pcsa(x)
            x = x + res1
        x = self.t_Conv2(x)
        for (aaconv, act, aaconv1, pcsa) in self.blocks2:
            res2 = x
            x = aaconv(x)
            x = act(x)
            x = aaconv1(x)
            x = pcsa(x)
            x = x + res2
        x = self.t_Conv3(x)
        x = x+x_shallow_features
        x = self.cattn(x)
        x = self.Convd(x)
        x = self.Convd_out(x)
        out = x + x_original_features
        # out=x
        return out


if __name__ == '__main__':
    net = AACNet(in_dim=305, out_dim=305, kernel_size=3, padding=1, stride=1, num_blocks=5).cuda()
    # net = AACNet(in_dim=7, out_dim=7, kernel_size=3, padding=1, stride=1, num_blocks=5).cuda()
    # net = AACNet(in_dim=4, out_dim=4, kernel_size=3, padding=1, stride=1, num_blocks=5).cuda()

    device = torch.device('cpu')
    net.to(device)
    # summary(net.cuda(), (305, 64, 64))


#=======================================================================================================================
# if __name__ == '__main__':
#     x = torch.randn(4, 305, 64, 64).cuda()
#     test_kernel_padding = [(3,1), (5,2), (7,3), (9,4),(11,5) ]
#     # mcplb = MCPLB(in_dim=150, out_dim=150, kernel_size=3, padding=1, stride=1, num_blocks=5).cuda()
#     mcplb = AACNet(in_dim=305, out_dim=305, kernel_size=3, padding=1, stride=1, num_blocks=5).cuda()
#     out = mcplb(x)
#     # summary(mcplb.cuda(), (150, 64, 64))
