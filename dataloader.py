import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import scipy.io as io
import torchvision
from torchvision import transforms as T
import numpy as np
import cv2
def MyLoader(path,type):
    if type=='img':
        return Image.open(path).convert('RGB')
    elif type=='npy':
        return np.load(path)
    elif type=='floor':
        return np.array(cv2.imread(path, cv2.IMREAD_GRAYSCALE))
    elif type == 'label':
        return np.array(Image.open(path)).astype(int)


class UISdataset_MM(Dataset):
    def __init__(self,txt,transform=None, target_transform=None, loader=MyLoader):
        with open(txt,'r') as fh:
            file=[]
            for line in fh:
                line=line.strip('\n')
                # line=line.rstrip().split('')
                # words=line.split()
                labelpath = line
                imapath = line.replace('label', 'img').replace('_all', '')
                floorpath = line.replace('label', 'floor').replace('_all', '').replace('jpg', 'npy')
                floorAreapath = line.replace('label', 'floorArea').replace('_all', '').replace('jpg', 'npy')
                superobjectpath = line.replace('label', 'object').replace('_all', '').replace('jpg', 'npy')

                # print(imapath, labelpath)
                file.append((imapath, floorpath, floorAreapath, superobjectpath, labelpath)) # 路径1 路径2 路径3 路径4 路径5 标签


        self.file=file
        self.transform=transform
        self.target_transform=target_transform
        self.loader=loader


    def __getitem__(self,index):

        img, floor, floorAreaPath, superobjectpath, label = self.file[index]

        img = self.loader(img,type='img')
        # print(np.array(img).shape)
        # msi=self.loader(lrs,type='msi')
        # hpi_f=self.loader(hpi,type='img')
        # label = self.loader(label, type='img')
        # # pois_f = self.loader(pois, type='vector')
        floor = self.loader(floor,type='npy')
        # print(floor[10, 10])
        floorArea = self.loader(floorAreaPath,type='npy')
        superobject_mask = self.loader(superobjectpath,type='npy')
        # instance_mask[instance_mask>0] = 1
        # instance_mask[instance_mask<=0] = 0
        # # print(floorArea[10, 10])
        label = self.loader(label,type='label')
        # if np.any(np.array(label) == 1):
        #     scene_label = 1
        # else:
        #     scene_label = 0
        # instance_label = instance_mask  # 0 / 1 / 2
        # # instance_label[instance_label<2] = 1
        # # print(label_f)

        if self.transform is not None:
            img=self.transform(img)
            # msi=torch.from_numpy(msi*1.0)
            # hpi_f=self.transform(hpi_f)
            label = self.transform(label)
            # pois_f=torch.from_numpy(pois_f)
            floor=self.transform(floor)
            floorArea=self.transform(floorArea)
            # superobject_mask=self.transform(superobject_mask)
            MMdata = torch.cat((floor, floorArea),0)

            # scene_label = self.transform(scene_label)
            # instance_label = self.transform(instance_label)
            # label_f=self.transform(label_f)
        # print(hrs_f.shape, msi.shape,hpi_f.shape,sv_f.shape,pois_f.shape,floor_f.shape, label_f.shape)

        # return hrs_f, msi,hpi_f,sv_f,pois_f,floor_f, label_f
        return img, MMdata, label

    def __len__(self):
        return len(self.file)


if __name__ == "__main__":
    test_transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    test_dataset=UISdataset_MM(txt='.\\data\\testSSOUISdataset_all.txt',transform=test_transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False,pin_memory=True)
    for step,(img, MMdata, label) in enumerate(test_loader):
        print(img[:, :, 10, 10], MMdata[:, :, 10, 10], label[:, :, 10, 10])
