
import torchvision
import torch.nn as nn
import torch
import torch.nn.functional as F
from .utils import get_default_boxes, xy_from_cxcy, find_IoU, encoding_from_cxcy, convert_to_ratio

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

class SSD_Loss(nn.Module):
    def __init__(self, default_boxes = get_default_boxes(), threshold = 0.5, hard_negative_ratio = 3, alpha = 1):
        super(SSD_Loss,self).__init__()
        self.default_boxes = default_boxes
        self.threshold = threshold
        self.hard_negative_ratio = hard_negative_ratio
        self.alpha = alpha
        self.cross_entrophy = nn.CrossEntropyLoss(reduction='none')
        self.smoothl1_loss = nn.SmoothL1Loss()

    def forward(self, predicted_boxes, predicted_scores, target_boxes, target_labels):
        batch_size = predicted_boxes.size(0)
        num_classes = predicted_scores.size(2)
        default_boxes_xy = xy_from_cxcy(self.default_boxes)
        num_default_boxes = self.default_boxes.size(0)
        target_boxes = convert_to_ratio(boxes=target_boxes)
    
        true_bboxes = torch.zeros((batch_size,num_default_boxes,4),dtype=torch.float).to(device)
        true_labels = torch.zeros((batch_size,num_default_boxes),dtype=torch.long).to(device)

        for i in range(batch_size):
            boxes = target_boxes[i]
            labels = target_labels[i]

            boxes_xy = xy_from_cxcy(boxes)
            overlap = find_IoU(boxes_xy, default_boxes_xy)
            boxes_with_obj, obj_idx = overlap.max(0)
            _, box_idx = overlap.max(1)
            obj_idx[box_idx] = torch.arange(overlap.size(0)).long().to(device)

            check_threshold = boxes_with_obj > self.threshold
            true_labels[i][check_threshold] = labels[obj_idx[check_threshold]]
            true_bboxes[i][check_threshold] = boxes[obj_idx[check_threshold]]
            true_bboxes[i] = encoding_from_cxcy(true_bboxes[i], self.default_boxes)

        positive = true_labels != 0
        num_hard_neg = (positive.sum(dim=1)*self.hard_negative_ratio).view(-1,1)
    
        # localization loss
        bbox_loss = self.smoothl1_loss(predicted_boxes[positive], true_bboxes[positive])

        # classification loss
        cls_loss_all = self.cross_entrophy(predicted_scores.view(-1,num_classes), true_labels.view(-1))
        cls_loss_all = cls_loss_all.view(batch_size,-1)
        cls_loss_pos = cls_loss_all[positive]
        
        cls_loss_neg = cls_loss_all.clone()
        cls_loss_neg[positive] = 0.
        cls_loss_neg , _ = cls_loss_neg.sort(dim=1, descending=True)
        cls_loss_idx = torch.LongTensor(range(num_default_boxes)).unsqueeze(0).expand_as(cls_loss_neg).to(device)
        hard_neg_idx = cls_loss_idx < num_hard_neg
        cls_loss_hard_neg = cls_loss_neg[hard_neg_idx]

        cls_loss = (cls_loss_pos.sum() + cls_loss_hard_neg.sum())/positive.sum().float()
        
        # total loss
        return cls_loss + self.alpha*bbox_loss