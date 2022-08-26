from platform import libc_ver
from typing import overload
import torch.nn as nn
import torch
import torch.nn.functional as F
from math import sqrt
from tqdm.auto import tqdm


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
                            extra_scale = sqrt(scales_list[k]*scales_list[k+1])
                        else:
                            extra_scale = 1
                        default_boxes.append([cx, cy, extra_scale, extra_scale])
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
    ## loss.py에서 true_bboxes가 -inf가 뜰 수 있음
    epsilon = 1e-7
    encoding_box = torch.cat([(cxcy[:,:2]-default_boxes[:,:2])/default_boxes[:,2:]*10, 5*torch.log((cxcy[:,2:])/(default_boxes[:,2:]))],1)
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


def object_detection(global_predicted_locs, global_predicted_scores, num_classes = 10, min_score = 0.05, max_overlap = 0.45, num_object_limit = 40):
    default_boxes = get_default_boxes()

    global_img_boxes = []
    global_img_labels = []
    global_img_scores = []
    for idx in tqdm(range(len(global_predicted_locs))):
        predicted_locs = global_predicted_locs[idx]
        predicted_scores = global_predicted_scores[idx]
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
                class_score = predicted_score[:,c]
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

            if img_boxes.size(0) > num_object_limit:
                img_scores , sorted_indice_for_limit = img_scores.sort(dim=0, descending = True)
                img_scores = img_scores[:num_object_limit]
                img_labels = img_labels[sorted_indice_for_limit][:num_object_limit]
                img_boxes = img_boxes[sorted_indice_for_limit][:num_object_limit]

            all_img_boxes.append(img_boxes)
            all_img_labels.append(img_labels)
            all_img_scores.append(img_scores)
        global_img_boxes.append(all_img_boxes)
        global_img_labels.append(all_img_labels)
        global_img_scores.append(all_img_scores)
    return global_img_boxes, global_img_labels, global_img_scores


def convert_to_ratio(image=False,boxes=[0,0,1,1]):
    if image is not False:
        h,w = image.size(1), image.size(2)
    else:
        h,w = 300,300
    ratio_boxes = []
    for box in boxes:
        ratio_boxes.append(torch.cat([box[...,:1]/w, box[...,1:2]/h, box[...,2:3]/w, box[...,3:4]/h], dim=-1))
    return ratio_boxes

def cal_mAP50(all_img_boxes, all_img_labels, all_img_scores, all_target_boxes, all_target_labels, num_classes = 10):
    all_precision = dict()
    all_recall = dict()
    mAP50_dict = dict()

    for i in range(len(all_img_boxes)):
        batch_size = len(all_img_boxes[i])

        for batch in range(batch_size):
            img_boxes = all_img_boxes[i][batch]
            img_labels = all_img_labels[i][batch]
            img_scores = all_img_scores[i][batch]
            target_boxes = all_target_boxes[i][batch]
            target_labels = all_target_labels[i][batch]

            for c in range(1,num_classes+1):
                target_idx = target_labels==c
                if target_idx.sum().item()==0:
                    continue
                class_idx = img_labels==c
                if class_idx.sum().item() == 0:
                    continue
                class_boxes = img_boxes[class_idx]
                class_scores = img_scores[class_idx]
                class_target_boxes = target_boxes[target_idx]

                class_boxes_xy = xy_from_cxcy(class_boxes)
                class_target_boxes_xy = xy_from_cxcy(class_target_boxes)
                iou, iou_label = find_IoU(class_target_boxes_xy,class_boxes_xy).max(0)
                sorted_scores, sorted_idx = class_scores.sort(dim=0,descending=True)
                sorted_iou = iou[sorted_idx]

                gt = len(class_target_boxes)
                tp = torch.Tensor([0]).to(device)
                fp = torch.Tensor([0]).to(device)
                check_target_box = torch.Tensor([False]*len(class_target_boxes)).bool().to(device)

                for j in range(len(sorted_scores)):
                    count_target_box = check_target_box.sum().item()
                    target_label = iou_label[j]
                    check_target_box[target_label] = True
                    if sorted_iou[j] < 0.5 or check_target_box.sum().item()==count_target_box:
                        fp+=1
                    else:
                        tp+=1

                    precision = tp/(tp+fp)
                    recall = tp/gt

                    if j==0:
                        class_precision = precision
                        class_recall = recall
                    else:
                        class_precision = torch.cat([class_precision,precision])
                        class_recall = torch.cat([class_recall,recall])               

                if c not in all_precision:
                    all_precision[c] = class_precision
                    all_recall[c] = class_recall
                else:
                    all_precision[c] = torch.cat([all_precision[c],class_precision])
                    all_recall[c] = torch.cat([all_recall[c],class_recall])
            
    for c in range(1,num_classes+1):
        if c not in all_recall or c not in all_precision:
            mAP50_dict[c] = 0
            continue
        sorted_recall, sorted_recall_idx = all_recall[c].sort()
        sorted_precision = all_precision[c][sorted_recall_idx]

        class_mAP50 = 0
        p=0
        r=1
        for i in range(len(sorted_recall)):
            if sorted_precision[i]>p:
                class_mAP50 += p*(r-sorted_recall[i])
                p = sorted_precision[i]
                r = sorted_recall[i]
            if i == len(sorted_recall)-1:
                class_mAP50 += p*r
        mAP50_dict[c] = class_mAP50

    return mAP50_dict
