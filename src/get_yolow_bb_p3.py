import onnxruntime
import cv2
import numpy as np
import time
import os
import matplotlib.pyplot as plt
import pickle
import torch

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

labels = ["bottle", "can", "cup", "box"]



CONFIG = "/home/semanticnuc/Desktop/Tiago/TIAGo-RoCoCo/KG_Reasoning/vlm-grasping/config/"
IMAGES = "/home/semanticnuc/Desktop/Tiago/TIAGo-RoCoCo/KG_Reasoning/vlm-grasping/images/"

# YOLOW_PATH = CONFIG + "yolow/yolow-l.onnx"
YOLOW_PATH = CONFIG + "yolow/sort.onnx"
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

    def __init__(self, yolo_path = YOLOW_PATH) -> None:
        self.yolow = onnxruntime.InferenceSession(yolo_path)
        torch.cuda.is_available() 
        if not torch.cuda.is_available():
            self.yolow.set_providers(['CPUExecutionProvider'])

        self.name_of_input = self.yolow.get_inputs()[0].name
        self.input_shape = self.yolow.get_inputs()[0].shape
        self.type_of_input = self.yolow.get_inputs()[0].type


    def __call__(self, image):
        cropped_image = crop_image(image, self.input_shape[3])
        cropped_image = np.transpose(cropped_image, (2, 0, 1)) 
        cropped_image = cropped_image / 255
        cropped_image = np.expand_dims(cropped_image, axis=0)  # Add batch dimension
        if "float" in self.type_of_input:
            input_tensor = cropped_image.astype(np.float32)
        else:
            input_tensor = cropped_image.astype(np.uint8)

        output = self.yolow.run(None, {'images': input_tensor})
        bboxs = output[1][0]
        scores = output[2][0]
        labels_idx = output[3][0]

        return bboxs, scores, labels_idx

        
def main():

    images = os.listdir(IMAGE_DIR)
    images.sort()

    sam = VitSam(encoder_model=ENCODER_PATH, decoder_model=DECODER_PATH)
    yolow = YOLOW(yolo_path=YOLOW_PATH)

    for image_path in images:

        image = cv2.imread(IMAGE_DIR + image_path)
       
        masked_image = image.copy()
        image_with_bbox = image.copy()

        #YOLOW inference
        bboxs, scores, labels_idx = yolow(image)

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