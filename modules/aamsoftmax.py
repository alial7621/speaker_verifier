import torch, math
import torch.nn as nn
import torch.nn.functional as F


class AAMsoftmax(nn.Module):
    """
    Modified AAMsoftmax loss function copied from voxceleb_trainer: 
    https://github.com/clovaai/voxceleb_trainer/blob/master/loss/aamsoftmax.py
    """
    def __init__(self, n_class, m, s, device='cuda'):
        super(AAMsoftmax, self).__init__()
        self.device = device
        self.m = m
        self.s = s
        self.weight = nn.Parameter(torch.FloatTensor(n_class, 192).to(self.device), requires_grad=True)
        self.ce = nn.CrossEntropyLoss().to(self.device)
        nn.init.xavier_normal_(self.weight, gain=1)
        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)
        self.th = math.cos(math.pi - self.m)
        self.mm = math.sin(math.pi - self.m) * self.m

    def forward(self, x, label=None):
        x = x.to(self.device)
        label = label.to(self.device)

        cosine = F.linear(F.normalize(x), F.normalize(self.weight))
        sine = torch.sqrt((1.0 - cosine ** 2).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where((cosine - self.th) > 0, phi, cosine - self.mm)
        
        one_hot = torch.zeros_like(cosine).to(self.device)
        one_hot.scatter_(1, label.view(-1, 1), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output = output * self.s

        loss = self.ce(output, label)

        return loss
    