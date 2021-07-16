import torch
import torch.nn as nn


class SimpleTrain(nn.Module):
    def __init__(self, args):
        super(SimpleTrain, self).__init__()
        self.predict_size = args.predict_size
        self.price_a = args.price_a
        self.price_b = args.price_b

    def forward(self, output, target):
        # open_ = target[:, :, 0]
        # close_ = target[:, :, 3]
        # target_mask = (open_ * self.price_b[3] + close_ * self.price_b[0] + open_ * close_ > 0).long()
        # loss = self.cross_entropy(output, target)
        loss_up = torch.abs(output[target == 1] - 1).mean()
        loss_down = torch.abs(output[target == 0] + 1).mean()
        return loss_up + loss_down


class SimpleTest(nn.Module):
    def __init__(self, args):
        super(SimpleTest, self).__init__()
        self.price_a = args.price_a
        self.price_b = args.price_b

    def forward(self, output, target):
        # open_ = target[:, 1, 0]
        # close_ = target[:, 1, 3]
        # target_true = open_ * self.price_b[3] + close_ * self.price_b[0] + open_ * close_ > 0

        predict_positive = output > 0
        # print(target)
        # print(output)
        # print(predict_positive)

        true_positive = target[predict_positive].sum().item()
        false_positive = torch.logical_not(target)[predict_positive].sum().item()
        false_negative = torch.logical_not(target)[torch.logical_not(predict_positive)].sum().item()
        return true_positive, false_positive, false_negative


class DistanceTrain(nn.Module):
    def __init__(self, args):
        super(DistanceTrain, self).__init__()
        self.price_a = args.price_a
        self.price_b = args.price_b
        self.predict_size = args.predict_size
        self.loss_func = nn.L1Loss()

    def forward(self, output, target):
        open_ = target[:, :, 0]
        close_ = target[:, :, 3]
        target_mask = (open_ * self.price_b[3] + close_ * self.price_b[0] + open_ * close_ > 0).view(-1)
        target_mask_not = torch.logical_not(target_mask)

        up, down = output[0].view(-1), output[1].view(-1)

        loss_up = self.loss_func(up[target_mask], torch.zeros_like(up[target_mask]))
        loss_up += self.loss_func(up[target_mask_not], torch.ones_like(up[target_mask_not]))

        loss_down = self.loss_func(down[target_mask_not], torch.zeros_like(down[target_mask_not]))
        loss_down += self.loss_func(down[target_mask], torch.ones_like(down[target_mask]))

        loss = loss_up + loss_down
        return loss


class DistanceTest(nn.Module):
    def __init__(self, args):
        super(DistanceTest, self).__init__()
        self.price_a = args.price_a
        self.price_b = args.price_b

    def forward(self, output, target):
        open_ = target[:, 1, 0]
        close_ = target[:, 1, 3]
        target_true = open_ * self.price_b[3] + close_ * self.price_b[0] + open_ * close_ > 0

        up, down = output[0][:, 1], output[1][:, 1]
        predict_positive = up < down

        true_positive = target_true[predict_positive].long().sum().item()
        false_positive = torch.logical_not(target_true)[predict_positive].long().sum().item()
        false_negative = torch.logical_not(target_true)[torch.logical_not(predict_positive)].long().sum().item()
        return true_positive, false_positive, false_negative


class CosineEmbeddingTrain(nn.Module):
    def __init__(self, args):
        super(CosineEmbeddingTrain, self).__init__()
        self.price_a = args.price_a
        self.price_b = args.price_b
        self.predict_size = args.predict_size

    def forward(self, output, target):
        open_ = target[:, :, 0]
        close_ = target[:, :, 3]
        target_mask = (open_ * self.price_b[3] + close_ * self.price_b[0] + open_ * close_ > 0).view(-1)

        up, down = output[0].view(-1), output[1].view(-1)

        loss_up = 1 - up[target_mask].mean() + torch.clamp(up[torch.logical_not(target_mask)], min=0).mean()
        loss_down = 1 - down[torch.logical_not(target_mask)].mean() + torch.clamp(down[target_mask], min=0).mean()
        loss = loss_up + loss_down
        return loss


class CosineEmbeddingTest(nn.Module):
    def __init__(self, args):
        super(CosineEmbeddingTest, self).__init__()
        self.price_a = args.price_a
        self.price_b = args.price_b

    def forward(self, output, target):
        open_ = target[:, :, 0] / self.price_b[0] + self.price_a[0]
        close_ = target[:, :, 3] / self.price_b[3] + self.price_a[3]
        o_c = (open_ * close_).view(-1)

        up, down = output[0].view(-1), output[1].view(-1)

        earnings_rate = o_c[up > down].log().sum().item()
        max_rate = o_c[o_c > 1].log().sum().item()
        average_rate = o_c.log().sum().item()

        return earnings_rate, max_rate, average_rate


class DistributeTrainLoss(nn.Module):
    def __init__(self, args):
        super(DistributeTrainLoss, self).__init__()
        self.gamma = args.gamma
        self.atoms = args.atoms
        self.predict_size = args.predict_size
        self.linspace = torch.linspace(-1, 1, args.atoms).view(1, 1, 1, 1, -1).to(args.device)

    def forward(self, output, price_f):
        price_f = price_f.unsqueeze(2)
        price_f_list = list()
        for i in range(self.predict_size):
            s, e = i + 1, -(self.predict_size - i - 1)
            if e == 0:
                price_f_list.append(price_f[:, s:])
            else:
                price_f_list.append(price_f[:, s:e])
        target = torch.cat(price_f_list, dim=2)

        predict_exp = torch.exp(output)
        predict_distribute = predict_exp / predict_exp.sum(-1).unsqueeze(-1)
        with torch.no_grad():
            predict_value = (self.linspace * predict_distribute).sum(-1)
            gap_value = target - predict_value
            target_distribute = self.calculate_target(predict_distribute, gap_value)

        loss = -torch.sum(target_distribute[:, self.predict_size:] * torch.log(predict_distribute[:, self.predict_size:] + 1e-8))
        target_size = target_distribute.size()
        return loss / (target_size[0] * (target_size[1] - self.predict_size) * target_size[2] * target_size[3])

    def calculate_target(self, predict_distribute, gap_value, Vmin=-1, Vmax=1):
        p_size = predict_distribute.size()
        batch_size = p_size[0] * p_size[1] * p_size[2] * p_size[3]

        support = torch.linspace(Vmin, Vmax, self.atoms).to(predict_distribute).unsqueeze(0)
        Tz = gap_value.view(-1).unsqueeze(1) + self.gamma * support
        Tz = Tz.clamp(min=Vmin, max=Vmax)
        b = (Tz - Vmin) / ((Vmax - Vmin) / (self.atoms - 1))
        l, u = b.floor().to(torch.int64), b.ceil().to(torch.int64)
        l[(u > 0) * (l == u)] -= 1
        u[(l < (self.atoms - 1)) * (l == u)] += 1

        pns_a = predict_distribute.clone().detach().view(-1, self.atoms)
        m = torch.zeros_like(pns_a)
        offset = torch.linspace(0, (batch_size - 1) * self.atoms, batch_size)
        offset = offset.unsqueeze(1).expand(batch_size, self.atoms).to(predict_distribute).to(torch.int64)

        m.view(-1).index_add_(0, (l + offset).view(-1), (pns_a * (u.float() - b)).view(-1))
        m.view(-1).index_add_(0, (u + offset).view(-1), (pns_a * (b - l.float())).view(-1))
        m = m.view(p_size[0], p_size[1], p_size[2], p_size[3], self.atoms)
        return m


class DistributeTestLoss(nn.Module):
    def __init__(self, args):
        super(DistributeTestLoss, self).__init__()
        self.atoms = args.atoms
        self.price_a = args.price_a
        self.price_b = args.price_b
        self.predict_size = args.predict_size
        self.linspace = torch.linspace(-1, 1, args.atoms).view(1, 1, 1, 1, -1).to(args.device)

    def forward(self, output, price_f):
        predict_exp = torch.exp(output)
        predict_distribute = predict_exp / predict_exp.sum(-1).unsqueeze(-1)
        predict_value = (self.linspace * predict_distribute).sum(-1)

        price_f = price_f.unsqueeze(2)
        price_f_list = list()
        for i in range(self.predict_size):
            s, e = i + 1, -(self.predict_size - i - 1)
            if e == 0:
                price_f_list.append(price_f[:, s:])
            else:
                price_f_list.append(price_f[:, s:e])
        target = torch.cat(price_f_list, dim=2)

        open_ = predict_value[:, :, :, 0] / self.price_b[0] + self.price_a[0]
        close_ = predict_value[:, :, :, 3] / self.price_b[3] + self.price_a[3]
        predict_result = (open_ * close_).view(-1)

        open_ = target[:, :, :, 0] / self.price_b[0] + self.price_a[0]
        close_ = target[:, :, :, 0] / self.price_b[3] + self.price_a[3]
        target_result = (open_ * close_).view(-1)

        earnings_rate = target_result[predict_result > 1].log().sum().item()
        opportunity_rate = target_result[predict_result < 1].log().sum().item()
        average_rate = target_result.log().sum().item()

        return earnings_rate, opportunity_rate, average_rate
