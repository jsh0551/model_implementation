import torch
from torch.utils.data import Dataset
import cv2
import numpy as np

class SSD_Dataset(Dataset):
    def __init__(self, json_file, path = '../.data/', train = True, transform=None):
        self.transform = transform
        super(SSD_Dataset, self).__init__()
        self.anns = json_file['annotations']
        self.image_infos = json_file['images']

        if train:
            self.path = path + 'train/'
        else:
            self.path = path + 'val/'

        self.transform = transform
        self.image_dict = dict()
        for img_info in json_file['images']:
            self.image_dict[img_info['id']] = img_info['file_name']

    def __getitem__(self, index):
        image_id = self.image_infos[index]['id']
        bboxes = []
        labels = []
        image = cv2.imread(self.path + self.image_dict[image_id])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        for ann in self.anns:
            if ann['image_id'] == image_id:
                bboxes.append(ann['bbox'])
                labels.append(ann['category_id'])

        if self.transform:
            transformed = self.transform(image=image,bboxes=bboxes,labels=labels)

            image = transformed['image']
            bboxes = transformed['bboxes']
            labels = transformed['labels']

        bboxes = torch.FloatTensor(bboxes)
        try:
            output_bboxes = torch.cat((bboxes[:,:2]+bboxes[:,2:]/2,bboxes[:,2:]),1)
        except IndexError:
            print("index error")
            print(bboxes.size())
        return torch.FloatTensor(image).permute(2,0,1), output_bboxes, torch.LongTensor(labels)

    def __len__(self):
        return len(self.image_dict)

    def collate_fn(self, batch):
        """
        Since each image may have a different number of objects, we need a collate function (to be passed to the DataLoader).
        This describes how to combine these tensors of different sizes. We use lists.
        Note: this need not be defined in this Class, can be standalone.
        :param batch: an iterable of N sets from __getitem__()
        :return: a tensor of images, lists of varying-size tensors of bounding boxes, labels, and difficulties
        """ 

        images = list()
        boxes = list()
        labels = list()


        for b in batch:
            images.append(b[0])
            boxes.append(b[1])
            labels.append(b[2])

        images = torch.stack(images, dim=0)

        return images,boxes,labels  # tensor (N, 3, 300, 300), 3 lists of N tensors each