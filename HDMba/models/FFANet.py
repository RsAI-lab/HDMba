import torch.nn as nn
import torch
# from torchsummary import summary
# import torchsummary


class PALayer(nn.Module):
    def __init__(self, channel):
        super(PALayer, self).__init__()
        self.pa = nn.Sequential(
                nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 8, 1, 1, padding=0, bias=True),
                nn.Sigmoid()
        )
    def forward(self, x):
        y = self.pa(x)
        return x * y


class CALayer(nn.Module):
    def __init__(self, channel):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
                nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 8, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.ca(y)
        return x * y


class Block(nn.Module):
    def __init__(self, dim):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(dim, dim, 3, padding=1, bias=True)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(dim, dim, 3, padding=1, bias=True)
        self.calayer = CALayer(dim)
        self.palayer = PALayer(dim)

    def forward(self, x):

        res = self.act1(self.conv1(x))
        res = res+x
        res = self.conv2(res)
        res = self.calayer(res)
        res = self.palayer(res)
        res += x
        return res


class GroupRLKAs(nn.Module):
    def __init__(self, dim):
        super(GroupRLKAs, self).__init__()
        self.g1 = Block(dim)
        self.g2 = Block(dim)
        self.g3 = Block(dim)

    def forward(self, x):
        y1 = self.g1(x)
        y2 = self.g1(y1)
        y3 = self.g1(y2)
        return y3, torch.cat([y1, y2, y3], dim=1)


class FFB(nn.Module):
    def __init__(self, dim):
        super(FFB, self).__init__()
        self.conv0 = nn.Conv2d(dim, 32, 1, bias=False)
        # self.activation0 = nn.ReLU()
        self.conv1 = nn.Conv2d(32, 32, 3, padding=1, bias=False)

    def forward(self, x):
        x = self.conv0(x)
        # x = self.activation0(x)
        x = self.conv1(x)
        return x


class FFANet(nn.Module):
    def __init__(self):
        super(FFANet, self).__init__()
        # 初始特征提取层
        # self.conv0 = nn.Conv2d(305, 32, 3, padding=1, bias=False)
        self.conv0 = nn.Conv2d(7, 32, 3, padding=1, bias=False)
        #self.conv0 = nn.Conv2d(4, 32, 3, padding=1, bias=False)
        self.conv1 = nn.Conv2d(32, 32, 3, padding=1, bias=False)

        # 中间的块
        self.g1 = GroupRLKAs(32)
        self.g2 = GroupRLKAs(32)
        self.g3 = GroupRLKAs(32)

        # 后面的块
        self.fusion = FFB(96*3)
        self.att1 = CALayer(32)
        self.att2 = PALayer(32)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1, bias=False)
        # self.conv3 = nn.Conv2d(32, 305, 3, padding=1, bias=False)
        self.conv3 = nn.Conv2d(32, 7, 3, padding=1, bias=False)
        #self.conv3 = nn.Conv2d(32, 4, 3, padding=1, bias=False)

    def forward(self, x):
        out1 = self.conv0(x)
        out2 = self.conv1(out1)

        R1, F1 = self.g1(out2)
        R2, F2 = self.g2(R1)
        R3, F3 = self.g2(R2)

        Fea = torch.cat([F1, F2, F3], dim=1)
        Fea = self.fusion(Fea)
        Fea = self.att1(Fea)
        Fea = self.att2(Fea)
        Fea = self.conv2(Fea)
        Fea = Fea + out1
        Fea = self.conv3(Fea)
        # Fea = Fea + x
        return Fea


if __name__ == "__main__":
    net = FFANet()

    # input_data = torch.rand(1, 305, 512, 512)
    # net = FFANet()
    # out = net(input_data)
    # print(out.shape)

    # summary(net, torch.rand(1, 305, 512, 512))

    device = torch.device('cpu')
    net.to(device)

    # torchsummary.summary(net.cuda(), (305, 512, 512))