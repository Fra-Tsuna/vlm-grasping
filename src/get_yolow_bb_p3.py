import cv2
import numpy as np
import time
import os
import matplotlib.pyplot as plt
import pickle
import torch

import torch
from mmengine.config import Config
from mmengine.dataset import Compose
from mmengine.runner import Runner
from mmengine.runner.amp import autocast
from mmyolo.registry import RUNNERS
from torchvision.ops import nms

from vitsam import VitSam
from vitsam import show_mask

# labels = [
#     "computer", "printer", "scanner", "keyboard", "mouse", "monitor", 
#     "desk", "chair", "binders", "folders", "pen", "pencil", "highlighter", 
#     "calendar", "bucket", "banner", "bottle", "remote control", 
#     "speaker", "robot", "book", "cables", "bag", "fire extinguisher", 
#     "device", "game controller", "tin can", "hammer", "screwdriver", 
#     "suitcase", "toolbox", "bag", "plank", "television", "wires", 
#     "electric plug", "screw", "nail", "box", "stool", "shelf"
# ]

labels = "(bottle, can, cup, box)"



CONFIG = "/home/michele/Desktop/Paper IROS/vlm-grasping/config/"
IMAGES = "/home/michele/Desktop/Paper IROS/vlm-grasping/images/"

# YOLOW_PATH = CONFIG + "yolow/yolow-l.onnx"
#YOLOW_PATH = CONFIG + "yolow/sort.onnx"
ENCODER_PATH = CONFIG + "efficientvitsam/l2_encoder.onnx"
DECODER_PATH = CONFIG + "efficientvitsam/l2_decoder.onnx"

# IMAGE_DIR = IMAGES + "scans/"
# YOLOW_OUTPUT_DIR = IMAGES + "yolow_output/"
# EVSAM_OUTPUT_DIR = IMAGES + "masked_output/"

IMAGE_DIR = IMAGES + "test_order/"
YOLOW_OUTPUT_DIR = IMAGE_DIR 
EVSAM_OUTPUT_DIR = IMAGE_DIR

DUMP = CONFIG + "dump_order/"

def crop_image(image_, target_size):
    # Read the image
    image = image_.copy()
    
    # Get current dimensions
    current_height, current_width = image.shape[:2]

    if current_height < current_width:
        crop_quantity = -(current_height-current_width)
        cropped_image = image[:,crop_quantity//2:-(crop_quantity//2)]
    else:
        crop_quantity = -(current_width-current_height)
        cropped_image = image[crop_quantity//2:-(crop_quantity//2),:]
    

    return cv2.resize(cropped_image, (target_size, target_size))


def convert_bb(old_coords):
    actual_image_size = (640,640)
    previous_image_size = (480,480)
    scale_width = previous_image_size[0] / actual_image_size[0]
    scale_height = previous_image_size[1] / actual_image_size[1]
    offset = (actual_image_size[0] - previous_image_size[0]) // 2
    x1, y1, x2, y2 = old_coords
    x1 = int(x1 * scale_width) +offset
    y1 = int((y1) * scale_height)
    x2 = int(x2 * scale_width) + offset
    y2 = int((y2) * scale_height) 
   
    return [x1,y1,x2,y2]


class YOLOW():

    def __init__(self):
        cfg = Config.fromfile(
            "src/yolo_world/yolo_world_l_t2i_bn_2e-4_100e_4x8gpus_obj365v1_goldg_train_lvis_minival.py"
        )
        cfg.work_dir = "."
        cfg.load_from = "src/yolo_world/yolow-v8_l_clipv2_frozen_t2iv2_bn_o365_goldg_pretrain.pth"
        cfg.__setattr__("log_level","WARNING")
        self.runner = Runner.from_cfg(cfg)
        self.runner.call_hook("before_run")
        self.runner.load_or_resume()
        pipeline = cfg.test_dataloader.dataset.pipeline
        self.runner.pipeline = Compose(pipeline)
        self.runner.model.eval()

    def set_class_name(self,objects):
        self.class_names = (objects)
        self.objects = objects.split(",")


    def __call__(self,input_image,max_num_boxes=100,score_thr=0.05,nms_thr=0.5):

        texts = [[t.strip()] for t in self.class_names.split(",")] + [[" "]]
        data_info = self.runner.pipeline(dict(img_id=0, img_path=input_image,
                                        texts=texts))

        data_batch = dict(
            inputs=data_info["inputs"].unsqueeze(0),
            data_samples=[data_info["data_samples"]],
        )

        with autocast(enabled=False), torch.no_grad():
            output = self.runner.model.test_step(data_batch)[0]
            self.runner.model.class_names = texts
            pred_instances = output.pred_instances

        keep_idxs = nms(pred_instances.bboxes, pred_instances.scores, iou_threshold=nms_thr)
        pred_instances = pred_instances[keep_idxs]
        pred_instances = pred_instances[pred_instances.scores.float() > score_thr]

        if len(pred_instances.scores) > max_num_boxes:
            indices = pred_instances.scores.float().topk(max_num_boxes)[1]
            pred_instances = pred_instances[indices]
        output.pred_instances = pred_instances

        pred_instances = pred_instances.cpu().numpy()

        xyxy=pred_instances['bboxes'],
        class_id=pred_instances['labels'],
        confidence=pred_instances['scores'] 

        return xyxy, confidence, class_id

        
def main():

    images = os.listdir(IMAGE_DIR)
    images.sort()

    sam = VitSam(encoder_model=ENCODER_PATH, decoder_model=DECODER_PATH)
    yolow = YOLOW()
    yolow.set_class_name(labels)
    for image_path in images:

        image = cv2.imread(IMAGE_DIR + image_path)
       
        masked_image = image.copy()
        image_with_bbox = image.copy()

        #YOLOW inference
        bboxs, scores, labels_idx = yolow(IMAGE_DIR + image_path)

        for i in range(len(bboxs)-1):
            
            score = scores[i]
            if score > 0.10:

                # EfficientViT SAM inference
                bbox = bboxs[i]

                bbox = convert_bb(bbox)
                x1,y1,x2,y2 = bbox

                label_idx = int(labels_idx[i])
                label = labels[label_idx]
                masks, _ = sam(masked_image, bbox)

                # Convert binary mask to 3-channel image
                overlay = masked_image 
                for mask in masks:
                    binary_mask = show_mask(mask)
                    with open(f'{DUMP}box{i}.pkl', 'wb') as f:
                        pickle.dump(binary_mask, f, protocol=2)
                    overlay = cv2.addWeighted(overlay, 1, binary_mask, 0.5, 0)

                label_idx = int(labels_idx[i])

                cv2.rectangle(image_with_bbox, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 1)
                cv2.putText(image_with_bbox, f"{label}: {score:.2f}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 1)
                cv2.imwrite(EVSAM_OUTPUT_DIR+image_path[:-4]+'_'+str(i)+'.jpg', overlay)

        cv2.imwrite(YOLOW_OUTPUT_DIR+image_path.replace(".jpg","_bb.png"), image_with_bbox)



if __name__ == '__main__':
    main()