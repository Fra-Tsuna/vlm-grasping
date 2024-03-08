
import cv2
import os
import time
import pickle

from models import VitSam, YOLOW
from models import show_mask



labels = "carton, table, cup, soda"


CONFIG = os.path.join(os.getcwd(),"config/")
IMAGES = os.path.join(os.getcwd(),"images/")

YOLOW_PATH = CONFIG + "yolow/"
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


        
def main():

    images = os.listdir(IMAGE_DIR)
    images.sort()

    dict_detections = {}

    sam = VitSam(encoder_model=ENCODER_PATH, decoder_model=DECODER_PATH)
    yolow = YOLOW(path=YOLOW_PATH)
    yolow.set_class_name(labels)
    for image_path in images:

        image = cv2.imread(IMAGE_DIR + image_path)
       
        masked_image = image.copy()
        image_with_bbox = image.copy()

        #YOLOW inference
        bboxs, scores, labels_idx = yolow(IMAGE_DIR + image_path)

        for i, (bbox,score,cls_id) in enumerate(zip(bboxs[0], scores, labels_idx[0])):
            
            if score > 0.10:

                # EfficientViT SAM inference

                # bbox = convert_bb(bbox)
                x1,y1,x2,y2 = bbox

                label = yolow.get_class_name(cls_id)
                masks, _ = sam(masked_image, bbox)
                if i not in dict_detections.keys():
                    dict_detections[i] = {'bbox':None,'label':None,'mask':None}
                    dict_detections[i]['bbox'] = bbox
                    dict_detections[i]['label'] = label

                # Convert binary mask to 3-channel image
                overlay = masked_image 
                for mask in masks:
                    binary_mask = show_mask(mask)
                    dict_detections[i]['mask'] = binary_mask
                    overlay = cv2.addWeighted(overlay, 1, binary_mask, 0.5, 0)

                cv2.rectangle(image_with_bbox, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 1)
                cv2.putText(image_with_bbox, f"{label}: {score:.2f}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 1)
                cv2.imwrite(EVSAM_OUTPUT_DIR+image_path[:-4]+'_'+str(i)+'.jpg', overlay)

        cv2.imwrite(YOLOW_OUTPUT_DIR+image_path.replace(".jpg","_bb.png"), image_with_bbox)
        
        with open(f'{DUMP}detection.pkl', 'wb') as f:
            pickle.dump(dict_detections, f, protocol=2)



if __name__ == '__main__':
    main()

    os.system("rm -rf config/yolow/logs")