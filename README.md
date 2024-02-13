# vlm-grasping

# YOLO-World Model
To download the YOLO-World ONNX model, visit the official huggingface [demo](https://huggingface.co/spaces/stevengrove/YOLO-World). 
If you want to use a set of labels different from the ones of the demo, write them in the box following the formatting instructions, and before exporting the model, test them on an image so that the new labels are correctly loaded in the model.
After the download, put the model in the ```config/yolow/``` directory.

# EfficientViT-SAM
To download the pre-trained weights and the encoder and decoder ONNX models follow the instruction on the official [repository](https://github.com/mit-han-lab/efficientvit).
After the download, put the models and weights in the ```config/efficientvitsam/``` directory
For our tests, we used the ```l2``` model.
