#!/usr/bin/env python

import rospy
from google.cloud import vision_v1 as vision

import io
import os
import base64

from PIL import Image

from webcolors import hex_to_rgb
from webcolors import CSS3_HEX_TO_NAMES
from sprockets.clients import memcached
DEBUG = False

PATH = os.path.abspath(__file__+'/../..')

OBJECTS_LIST = "/config/objects.txt"
MATERIALS_LIST = "/config/materials.txt"
COLORS_LIST = "/config/colors.txt"
SHAPES_LIST = "/config/shapes.txt"

REL_DICT ={('o','o'): "ObjIsRelatedTo",
           ('o','l'): "ObjOn",
           ('o','r'): "ObjInRoom",
           ('o','m'): "ObjHasMaterial",
           ('o','c'): "ObjHasColor",
           ('o','p'): "ObjHasShape"
           }

GRAPH = "/graph/graph.txt"
JSON_FILE = '/config/second-modem-381513-1442ee488e5a.json'

def check_label_in_list(label):
    label = label.lower()
    label = label.replace(" ", "-")
    with open(PATH+MATERIALS_LIST, "r") as f:
        lines = f.readlines()
    with open(PATH+COLORS_LIST, "r") as f:
        lines += f.readlines()
    with open(PATH+SHAPES_LIST, "r") as f:
        lines += f.readlines()
    with open(PATH+OBJECTS_LIST, "r") as f:
        lines += f.readlines()
    
    label_type = None
    for line in lines:
        curr_line = line[:-1]
        if label == curr_line[:-2]:
            label_type = curr_line[-1]
            return (label, label_type)
        
    return (label, label_type)


def augment_graph(h,t,h_lab,t_lab):
    key = (h_lab, t_lab)
    rel = REL_DICT[key]
    full_rel = h+'.'+h_lab+' '+rel+' '+t+'.'+t_lab
    with open(PATH+GRAPH,'a') as f:
        f.write(full_rel+"\n")
    return


def extractData():
    #it should be in an init but since this is called from external script...
    # Activate Google vision API using service account key


    client = vision.ImageAnnotatorClient.from_service_account_json(PATH+JSON_FILE)

    pic = client_memch.get('image_n')
    if pic == None:
        # objects = []
        # client_memch.set('objs', objects)
        return
    
    im_pil= io.BytesIO(base64.b64decode(pic)).read()
    im_size=Image.open(io.BytesIO(base64.b64decode(pic)))

    query = {"image": {"content": im_pil},
            "features": [{"type_": "LABEL_DETECTION"},
                        {"type_": "OBJECT_LOCALIZATION"},
                        {"type_": "IMAGE_PROPERTIES"}],}

    response = client.annotate_image(query)
    objects = response.localized_object_annotations

    bounding_boxes = []
    object_names = []
    width, height = im_size.size

    for i in range(len(objects)):
        x = []
        y = []
        for vertex in objects[i].bounding_poly.normalized_vertices:
            x.append(vertex.x)
            y.append(vertex.y)
        
        global_vertices = [int(width*x[0]), 
                    int(height*y[0]), 
                    int(width*x[2]), 
                    int(height*y[2])]
    
        bounding_boxes.append(global_vertices)
        object_names.append(objects[i].name)

    list_of_objs_with_bb = [{"bbox": bbox, "label": label} for bbox, label in zip(bounding_boxes, object_names)]
  
    return list_of_objs_with_bb




os.environ['CD_MEMCACHED_SERVERS'] = '127.0.0.1:11211<:1000M>'
client_memch = memcached.Client('cloud_detector')
objects_with_bb = extractData()


client_memch.set('objs_with_bb', objects_with_bb)