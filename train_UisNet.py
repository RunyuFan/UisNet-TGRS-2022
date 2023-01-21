from dataloader import UISdataset_MM
import argparse
from torchvision.models import resnet50, resnext50_32x4d, densenet121
import pretrainedmodels
from pretrainedmodels.models import *
# from models.segformer import SegFormer
import torch
import torch.nn as nn
import os
import torchvision
from torch import nn, optim
from torch.utils import data
from torchvision import transforms
import time
from torch import nn, Tensor
from torch.nn import functional as F
from tabulate import tabulate
from model_fuse import SOST
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class OhemCrossEntropy(nn.Module):
    def __init__(self, ignore_label: int = 255, weight: Tensor = None, thresh: float = 0.7, aux_weights: list = [1, 0.4]) -> None:
        super().__init__()
        self.ignore_label = ignore_label
        self.aux_weights = aux_weights
        self.thresh = -torch.log(torch.tensor(thresh, dtype=torch.float))
        self.criterion = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_label, reduction='none')

    def _forward(self, preds: Tensor, labels: Tensor) -> Tensor:
        # preds in shape [B, C, H, W] and labels in shape [B, H, W]
        if preds.shape[-2:] != labels.shape[-2:]:
            preds = F.interpolate(preds, size=labels.shape[1:], mode='bilinear', align_corners=False)

        n_min = labels[labels != self.ignore_label].numel() // 16
        loss = self.criterion(preds, labels).view(-1)
        loss_hard = loss[loss > self.thresh]

        if loss_hard.numel() < n_min:
            loss_hard, _ = loss.topk(n_min)

        return torch.mean(loss_hard)

    def forward(self, preds, labels: Tensor) -> Tensor:
        if isinstance(preds, list):
            return sum([w * self._forward(pred, labels) for (pred, w) in zip(preds, self.aux_weights)])
        return self._forward(preds, labels)

class Dice(nn.Module):
    def __init__(self, delta: float = 0.5, aux_weights: list = [1, 0.4]):
        """
        delta: Controls weight given to FP and FN. This equals to dice score when delta=0.5
        """
        super().__init__()
        self.delta = delta
        self.aux_weights = aux_weights

    def _forward(self, preds: Tensor, targets: Tensor) -> Tensor:
        # preds in shape [B, C, H, W] and targets in shape [B, C, H, W]
        if preds.shape[-2:] != targets.shape[-2:]:
            preds = F.interpolate(preds, size=targets.shape[2:], mode='bilinear', align_corners=False)

        tp = torch.sum(targets*preds, dim=(2, 3))
        fn = torch.sum(targets*(1-preds), dim=(2, 3))
        fp = torch.sum((1-targets)*preds, dim=(2, 3))

        dice_score = (tp + 1e-6) / (tp + self.delta * fn + (1 - self.delta) * fp + 1e-6)
        dice_score = torch.sum(1-dice_score, dim=-1)

        # adjust loss to account for number of classes
        dice_score = dice_score / targets.shape[1]
        return dice_score.mean()

    def forward(self, preds, targets: Tensor) -> Tensor:
        if isinstance(preds, list):
            return sum([w * self._forward(pred, targets) for (pred, w) in zip(preds, self.aux_weights)])
        return self._forward(preds, targets)


def main(args):
    # Create model
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    train_txt = "./data/trainSSOUISdataset_all.txt"
    val_txt = "./data/valSSOUISdataset_all.txt"
    test_txt = "./data/testSSOUISdataset_all.txt"

    train_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()
                                                ])
    val_transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    test_transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

    train_dataset=UISdataset_MM(txt=train_txt,transform=train_transform)
    val_dataset=UISdataset_MM(txt=val_txt,transform=test_transform)
    test_dataset=UISdataset_MM(txt=test_txt,transform=test_transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader=torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,num_workers=8,pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,num_workers=8,pin_memory=True)

    print("Train numbers:{:d}".format(len(train_dataset)))
    print("val numbers:{:d}".format(len(val_dataset)))
    print("Test numbers:{:d}".format(len(test_dataset)))

    # model2 = MultiModalNet()
    model2 = SOST(2)

    print('model2 parameters:', sum(p.numel() for p in model2.parameters() if p.requires_grad))

    # model1 = model1.to(device)
    model2 = model2.to(device)
    # model3 = model3.to(device)
    cost2 = OhemCrossEntropy().to(device)
    # cost2 = nn.CrossEntropyLoss().to(device)
    # cost2 = Dice().to(device)
    # cost2 = nn.NLLLoss().to(device)
    # Optimization
    # optimizer1 = optim.Adam(model1.parameters(), lr=args.lr, weight_decay=1e-6)
    optimizer2 = optim.Adam(model2.parameters(), lr=args.lr, weight_decay=1e-6)
    # optimizer2 = optim.SGD([
    #             {"params":large_lr_layers},
    #             {"params":small_lr_layers,"lr":1e-3}
    #             ],lr = 1e-2,momentum=0.9)
    # params = [
    # {"params": large_lr_layers, "lr": 1e-2},
    # {"params": small_lr_layers, "lr": 1e-4},
    # ]
    # optimizer2 = optim.SGD(params, momentum=0.9)

    # optimizer3 = optim.Adam(model3.parameters(), lr=args.lr, weight_decay=1e-6)


    # best_acc_1 = 0.
    miou_max = 0.
    best_epoch = 0
    # best_acc_3 = 0.
    # alpha = 1
    for epoch in range(1, args.epochs + 1):
        # model1.train()
        model2.train()
        # model3.train()
        # start time
        start = time.time()
        index = 0
        for images, mmdata, labels in train_loader:
            images = images.to(device)
            mmdata = mmdata.to(device)
            # print(images.shape)
            labels = labels.to(device, dtype=torch.int64)
            # instance_label = instance_label.to(device, dtype=torch.int64)
            mmdata = mmdata.clone().detach().float()
            images = images.clone().detach().float()
            # labels = labels.clone().detach().Long()

            # Forward pass
            # outputs1 = model1(images)
            _1, _2, _3, outputs2 = model2(images, mmdata)
            # print(outputs2.shape)
            # outputs3 = model3(images)
            # loss1 = cost1(outputs1, labels)
            # print(outputs2.shape, labels.squeeze(1).shape, instance_out.shape, instance_label.squeeze(1).shape)
            loss2 = cost2(outputs2, labels.squeeze(1))
            # loss3 = cost3(outputs3, labels)

            # if index % 10 == 0:
                # print (loss)
            # Backward and optimize
            # optimizer1.zero_grad()
            optimizer2.zero_grad()
            # optimizer3.zero_grad()
            # loss1.backward()
            loss2.backward()
            # loss3.backward()
            # optimizer1.step()
            optimizer2.step()
            # optimizer3.step()
            index += 1


        if epoch % 1 == 0:
            end = time.time()
            # print("Epoch [%d/%d], Loss: %.8f, Time: %.1fsec!" % (epoch, args.epochs, loss1.item(), (end-start) * 2))
            print("Epoch [%d/%d], Loss: %.8f, Time: %.1fsec!" % (epoch, args.epochs, loss2.item(), (end-start) * 2))
            # print("Epoch [%d/%d], Loss: %.8f, Time: %.1fsec!" % (epoch, args.epochs, loss3.item(), (end-start) * 2))

            # model1.eval()
            model2.eval()
            # model3.eval()

            # classes = ('bareland', 'cropland', 'forest', 'impervious', 'shrub', 'water')
            classes = ['Formal', 'Informal'] # ('住宅区', '公共服务区域', '商业区', '城市绿地', '工业区')

            hist = torch.zeros(args.num_class, args.num_class).to(device)

            with torch.no_grad():
                for images, mmdata, labels in val_loader:
                    images = images.to(device)
                    mmdata = mmdata.to(device)
                    # print(images.shape)
                    labels = labels.to(device, dtype=torch.int64).squeeze(1)
                    mmdata = mmdata.clone().detach().float()
                    images = images.clone().detach().float()
                    # print(labels.shape)

                    # Forward pass
                    # outputs1 = model1(images)
                    _1, _2, _3, outputs2 = model2(images, mmdata)
                    # outputs3 = model3(images)
                    # print(outputs2.shape)
                    # loss1 = cost1(outputs1, labels)
                    preds = outputs2.softmax(dim=1).argmax(dim=1)
                    # print(preds.shape)

                    keep = labels != 1000
                    hist += torch.bincount(labels[keep] * args.num_class + preds[keep], minlength=args.num_class**2).view(args.num_class, args.num_class)

            ious = hist.diag() / (hist.sum(0) + hist.sum(1) - hist.diag())
            miou = ious[~ious.isnan()].mean().item()
            ious = ious.cpu().numpy().tolist()
            miou = miou * 100

            Acc = hist.diag() / hist.sum(1)
            mOA = hist.diag().sum() / hist.sum()

            table = {
                'Class': classes,
                'IoU': ious,
                'Acc': Acc,
                # 'mOA': mOA
            }

            print(tabulate(table, headers='keys'))
            print(f"\nOverall mIoU: {miou:.2f}")

        if  miou > miou_max:
            print('save new best miou', miou)
            torch.save(model2, os.path.join(args.model_path, 'UIS-Segformer-SOST-Tiny-100.pth'))
            miou_max = miou
            best_epoch = epoch
        # if acc_3 > best_acc_3:
        #     print('save new best acc_3', acc_3)
        #     torch.save(model3, os.path.join(args.model_path, 'AID-30-teacher-densenet121-%s.pth' % (args.model_name)))
        #     best_acc_3 = acc_3
    # print("Model save to %s."%(os.path.join(args.model_path, 'UFZ-teacher-model-%s.pth' % (args.model_name))))
    # print('save new best acc_1', best_acc_1)
    print('save new best miou', miou_max, best_epoch)
    # print('save new best acc_3', best_acc_3)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='train hyper-parameter')
    parser.add_argument("--num_class", default=2, type=int)
    parser.add_argument("--epochs", default=200, type=int)
    # parser.add_argument("--net", default='ResNet50', type=str)
    # parser.add_argument("--depth", default=50, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--batch_size", default=64, type=int)
    # parser.add_argument("--num_workers", default=2, type=int)
    parser.add_argument("--model_name", default='', type=str)
    parser.add_argument("--model_path", default='./model-UIS', type=str)
    parser.add_argument("--pretrained", default=False, type=bool)
    args = parser.parse_args()

    main(args)
