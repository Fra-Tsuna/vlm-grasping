import onnxruntime
import cv2
import numpy as np
import time
import os
import matplotlib.pyplot as plt

from vitsam import VitSam
from vitsam import show_mask

labels = [
    "computer", "printer", "scanner", "keyboard", "mouse", "monitor", 
    "desk", "chair", "binders", "folders", "pen", "pencil", "highlighter", 
    "calendar", "bucket", "banner", "bottle", "remote control", 
    "speaker", "robot", "book", "cables", "bag", "fire extinguisher", 
    "device", "game controller", "tin can", "hammer", "screwdriver", 
    "suitcase", "toolbox", "bag", "plank", "television", "wires", 
    "electric plug", "screw", "nail", "box", "stool", "shelf"
]


CONFIG = "/home/semanticnuc/Desktop/Tiago/TIAGo-RoCoCo/KG_Reasoning/vlm_grasping/config/"
IMAGES = "/home/semanticnuc/Desktop/Tiago/TIAGo-RoCoCo/KG_Reasoning/vlm_grasping/images/"

YOLOW_PATH = CONFIG + "yolow/yolow-l.onnx"
ENCODER_PATH = CONFIG + "efficientvitsam/l2_encoder.onnx"
DECODER_PATH = CONFIG + "efficientvitsam/l2_decoder.onnx"

IMAGE_DIR = IMAGES + "scans/"
YOLOW_OUTPUT_DIR = IMAGES + "yolow_output/"
EVSAM_OUTPUT_DIR = IMAGES + "masked_output/"

session = onnxruntime.InferenceSession(YOLOW_PATH)
session.set_providers(['CPUExecutionProvider'])
name_of_input = session.get_inputs()[0].name  
input_shape = session.get_inputs()[0].shape
type_of_input = session.get_inputs()[0].type 


images = os.listdir(IMAGE_DIR)
images.sort()

sam = VitSam(encoder_model=ENCODER_PATH, decoder_model=DECODER_PATH)

for image_path in images:
# for j in range(1):
    image = cv2.imread(IMAGE_DIR + image_path)
    # image = cv2.imread(IMAGE_DIR + "test.jpeg")
    masked_image = image.copy()
    masked_image = cv2.resize(masked_image, (input_shape[3], input_shape[2]))
    cv2.imwrite(EVSAM_OUTPUT_DIR + str("boh_")+image_path, masked_image)
    image = cv2.resize(image, (input_shape[3], input_shape[2]))

    image_with_bbox = image.copy()
 
    image = np.transpose(image, (2, 0, 1)) 
    image = image / 255
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    if "float" in type_of_input:
        input_tensor = image.astype(np.float32)
    else:
        input_tensor = image.astype(np.uint8)

    start_time = time.time()
    output = session.run(None, {'images': input_tensor})
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time} seconds")

    num_dects = output[0][0]
    bboxs = output[1][0]
    scores = output[2][0]
    labels_idx = output[3][0]

    for i in range(len(bboxs)-1):
        score = scores[i]
        if score > 0.25:
            bbox = bboxs[i]
            label_idx = int(labels_idx[i])
            label = labels[label_idx]
            masks, _ = sam(masked_image, bbox)

            # Convert binary mask to 3-channel image
            overlay = masked_image 
            for mask in masks:
                binary_mask = show_mask(mask)
                overlay = cv2.addWeighted(overlay, 1, binary_mask, 0.5, 0)

            label_idx = int(labels_idx[i])
            x1, y1, x2, y2 = bbox
            cv2.rectangle(image_with_bbox, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 1)
            cv2.putText(image_with_bbox, f"{label}: {score:.2f}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 1)

            cv2.imwrite(EVSAM_OUTPUT_DIR+image_path[:-4]+'_'+str(i)+'.jpg', overlay)
            # cv2.imwrite(EVSAM_OUTPUT_DIR+"test_"+str(i)+".jpg", overlay)

    cv2.imwrite(YOLOW_OUTPUT_DIR+image_path, image_with_bbox)
    # cv2.imwrite(YOLOW_OUTPUT_DIR+"test_"+str(i)+".jpg", image_with_bbox)
