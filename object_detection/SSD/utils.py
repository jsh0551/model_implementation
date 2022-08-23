import torch.nn as nn
import torch
import torch.nn.functional as F
from math import sqrt

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def get_default_boxes(ft_map_size=[38,19,10,5,3,1], num_boxes=[4,6,6,6,4,4], scales_list=[0.1,0.2,0.375,0.55,0.725,0.9]):
    default_boxes = []
    for k,size in enumerate(ft_map_size):
        for i in range(size):
            for j in range(size):
                cx = (i + 0.5)/size
                cy = (j + 0.5)/size

                max_ratio = num_boxes[k]//2
                scale = scales_list[k]
                for ratio in range(1,max_ratio+1):
                    if ratio == 1:
                        default_boxes.append([cx, cy, scale, scale])
                        if k < len(scales_list)-1:
                            extra_ratio = scales_list[k]*scales_list[k+1]
                        else:
                            extra_ratio = 1
                        default_boxes.append([cx, cy, scale*sqrt(extra_ratio), scale*sqrt(extra_ratio)])
                    else:
                        default_boxes.append([cx, cy, scale*sqrt(ratio), scale/sqrt(ratio)])
                        default_boxes.append([cx, cy, scale/sqrt(ratio), scale*sqrt(ratio)])


    default_boxes = torch.FloatTensor(default_boxes).to(device)
    default_boxes = default_boxes.clamp(0,1)
    return default_boxes

def cxcy_from_prediction(predicted_loc, default_boxes):
    cxcy = torch.cat([predicted_loc[:,:2]*default_boxes[:,2:]/10 + default_boxes[:,:2]
                     ,torch.exp(predicted_loc[:,2:]/5)*default_boxes[:,2:]],1)
    return cxcy


def xy_from_cxcy(cxcy):
    xy = torch.cat([cxcy[:,:2]-cxcy[:,2:]/2, cxcy[:,:2]+cxcy[:,2:]/2], 1)
    return xy

def cxcy_from_xy(xy):
    cxcy = torch.cat([(xy[:,2:]+xy[:,:2])/2, xy[:,2:]-xy[:,:2]], 1)
    return cxcy

def encoding_from_cxcy(cxcy, default_boxes):
    encoding_box = torch.cat([(cxcy[:,:2]-default_boxes[:,:2])/default_boxes[:,2:]*10, 5*torch.log(cxcy[:,2:]/default_boxes[:,2:])],1)
    return encoding_box

def find_intersection(xy1,xy2):
    lower_bound = torch.max(xy1[:,:2].unsqueeze(1), xy2[:,:2].unsqueeze(0))
    upper_bound = torch.min(xy1[:,2:].unsqueeze(1), xy2[:,2:].unsqueeze(0))
    intersection_wh = torch.clamp(upper_bound - lower_bound,min=0)
    return intersection_wh[...,0]*intersection_wh[...,1]

def find_IoU(xy1,xy2):
    area1 = (xy1[:,2] - xy1[:,0]) * (xy1[:,3] - xy1[:,1])
    area2 = (xy2[:,2] - xy2[:,0]) * (xy2[:,3] - xy2[:,1])
    intersection = find_intersection(xy1, xy2)
    union = area1.unsqueeze(1) + area2.unsqueeze(0) - intersection
    return intersection / union

def object_detection(predicted_locs, predicted_scores, num_classes = 10, min_score = 0.01, max_overlap = 0.45, num_limit_obj = 200):
    default_boxes = get_default_boxes()
    batch_size = predicted_locs.size(0)
    all_img_boxes = []
    all_img_labels = []
    all_img_scores = []
    predicted_scores = predicted_scores.softmax(dim=-1)

    for i in range(batch_size):
        predicted_xy = xy_from_cxcy(cxcy_from_prediction(predicted_locs[i], default_boxes))
        predicted_score = predicted_scores[i]

        img_boxes = []
        img_labels = []
        img_scores = []

        for c in range(1,num_classes+1):
            class_score = predicted_score[:,c].softmax(dim=1)
            check_min_score = class_score > min_score
            if check_min_score.sum().item() == 0:
                continue
            
            class_score = class_score[check_min_score]
            class_predicted_xy = predicted_xy[check_min_score]

            class_score, sorted_indice = class_score.sort(dim=0, descending=True)
            class_predicted_xy = class_predicted_xy[sorted_indice]

            iou = find_IoU(class_predicted_xy, class_predicted_xy)

            ## NMS
            need_suppress = torch.zeros(iou.size(0),dtype=torch.uint8).to(device)
            for box in range(iou.size(0)):
                if need_suppress[box] == 1:
                    continue
                need_suppress = torch.max(need_suppress, iou[box] > max_overlap)
                need_suppress[box] = 0

            img_boxes.append(class_predicted_xy[(1-need_suppress).bool()])
            img_labels.append(torch.LongTensor((1-need_suppress).sum().item()*[c]).to(device))
            img_scores.append(class_score[(1-need_suppress).bool()])

        if len(img_boxes) == 0:
            img_boxes.append(torch.FloatTensor([[0, 0, 1, 1]]).to(device))
            img_labels.append(torch.LongTensor([0]).to(device))
            img_scores.append(torch.FloatTensor([0]).to(device))

        img_boxes = torch.cat(img_boxes,dim=0)
        img_labels = torch.cat(img_labels,dim=0)
        img_scores = torch.cat(img_scores,dim=0)

        if img_boxes.size(0) > num_limit_obj:
            img_scores , sorted_indice_for_limit = img_scores.sort(dim=0, descending = True)
            img_scores = img_scores[:num_limit_obj]
            img_labels = img_labels[sorted_indice_for_limit][:num_limit_obj]
            img_boxes = img_boxes[sorted_indice_for_limit][:num_limit_obj]

        all_img_boxes.append(img_boxes)
        all_img_labels.append(img_labels)
        all_img_scores.append(img_scores)

    return all_img_boxes, all_img_labels, all_img_scores