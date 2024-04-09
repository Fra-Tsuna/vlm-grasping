<div align="center">
<img src="/assets/logo3.png" width=60%>
<br>
<a href="https://www.linkedin.com/in/fra-arg/">Francesco Argenziano</a><sup><span>1</span></sup>,
<a href="https://scholar.google.com/citations?user=sk3SpmUAAAAJ&hl=it&oi=ao/">Michele Brienza</a><sup><span>1</span></sup>,
<a href="https://scholar.google.com/citations?user=Y8LuLfoAAAAJ&hl=it&oi=ao">Vincenzo Suriani</a><sup><span>2</span></sup>,
<a href="https://scholar.google.com/citations?user=xZwripcAAAAJ&hl=it&oi=ao">Daniele Nardi</a><sup><span>1</span></sup>,
<a href="https://scholar.google.com/citations?user=_90LQXQAAAAJ&hl=it&oi=ao">Domenico D. Bloisi</a><sup><span>3</span></sup>
</br>

<sup>1</sup> Department of Computer, Control and Management Engineering, Sapienza University of Rome, Rome, Italy,
<sup>2</sup> School of Engineering, University of Basilicata, Potenza, Italy,
<sup>3</sup> International University of Rome UNINT, Rome, Italy
<div>

[![arxiv paper](https://img.shields.io/badge/Project-Website-blue)](https://sites.google.com/diag.uniroma1.it/empower/home)
[![arxiv paper](https://img.shields.io/badge/arXiv-TBA-red)](https://sites.google.com/diag.uniroma1.it/empower/home)
[![license](https://img.shields.io/badge/License-Apache_2.0-yellow)](LICENSE)

</div>
</div>

## Prerequisites
First, clone this repo
```
https://github.com/Fra-Tsuna/vlm-grasping.git
```


## Model checkpoints
### Yolo-World
To download the YOLO-World ONNX model, visit the official huggingface [demo](https://huggingface.co/spaces/stevengrove/YOLO-World). 
If you want to use a set of labels different from the ones of the demo, write them in the box following the formatting instructions, and before exporting the model, test them on an image so that the new labels are correctly loaded in the model.
After the download, put the model in the ```config/yolow/``` directory.
### EfficientViT-SAM
To download the pre-trained weights and the encoder and decoder ONNX models follow the instruction on the official [repository](https://github.com/mit-han-lab/efficientvit).
After the download, put the models and weights in the ```config/efficientvitsam/``` directory
For our tests, we used the ```l2``` model.
