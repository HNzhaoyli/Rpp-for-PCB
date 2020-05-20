import torchvision.models as models
import torch.nn as nn
import torch
from torch.autograd import Variable
from torch.nn import init
from torch.nn import functional as F

'''
confient = nn.Sequential(nn.Conv2d(32,6,1,1,padding=0),#六个线性分类器
                         nn.Softmax(dim=1))
GAP = nn.AdaptiveAvgPool2d((1,1))
x = torch.rand(8,32,12,12)
y = confient(x) # 大小(8,6,12,12)即6张概率图
confs={}
part={}
for i in range(y.size(1)):
    confs[i] = torch.unsqueeze(y[:,i,:,:],1)
    confs[i] = confs[i].expand(x.size(0), x.size(1), x.size(2), x.size(3))
    tmp = torch.mul(x,confs[i])
    part[i] = torch.squeeze(GAP(tmp),3)

rst = part[0]
for i in range(1,y.size(1)):
    rst = torch.cat([rst,part[i]],dim=2)

print(rst.size())
'''
class RPP(nn.Module):
    def __init__(self, in_channels, num_parts):
        super(RPP, self).__init__()
        self.confient = nn.Sequential(nn.Conv2d(in_channels,num_parts,1,1,padding=0),#六个线性分类器
                         nn.Softmax(dim=1))
        self.GAP = nn.AdaptiveAvgPool2d((1,1))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif isinstance(m, nn.BatchNorm2d):
                init.normal_(m.weight.data, 1.0, 0.02)
                init.constant_(m.bias.data, 0.0)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
                init.constant_(m.bias.data, 0.0)
    def forward(self, x):
        y = self.confient(x)
        confs = {}
        part = {}
        for i in range(y.size(1)):
            confs[i] = torch.unsqueeze(y[:, i, :, :], 1)
            confs[i] = confs[i].expand(x.size(0), x.size(1), x.size(2), x.size(3))
            tmp = torch.mul(x, confs[i])
            part[i] = torch.squeeze(self.GAP(tmp), 3)

        out = part[0]
        for i in range(1, y.size(1)):
            out = torch.cat([out, part[i]], dim=2)
        return out

if __name__ == '__main__':
    inputs = Variable(torch.randn(16, 4, 32, 32))
    Rpp = RPP(4,6)
    out = Rpp(inputs)
    print(out.shape)






