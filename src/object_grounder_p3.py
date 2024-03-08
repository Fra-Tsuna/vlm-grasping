from src.init import Loader
import numpy as np
import pickle

from sklearn.metrics.pairwise import cosine_similarity

print(Loader().get_yolow_model())

with open(config.AGENTS+'results_multi_order_the_rgb.pkl', 'rb') as file:
    data = pickle.load(file)

object_relations = data['enviroment_info'].split('\n')
relation_list = []
print(object_relations)
planning = data['planning_agent_info']

planning = planning[planning.index("1"):]
print(planning)

relation_list = []

def split_word(words):
    splitted_word = []
    doc = config.nlp(words)
    for token in doc:
        print(token.text, token.pos_, token.dep_)
        if token.pos_ == "NOUN" and token.dep_ in ["dobj","ROOT","nsubj"] or token.pos_ == "VERB":
            splitted_word.append(token.text)
    return splitted_word

def compare_two_words(list1, list2):
    word1 = list1.copy()
    word2 = list2.copy()
    min_1 = len(word1)
    min_2 = len(word2)
    sim_word = []
    if min_1 > min_2:
        for index in range(min_1):
            trovato = False
            sim = 0
            min_2 = len(word2)
            for index_2 in range(min_2):
                sim = (config.wv.similarity(word1[index], word2[index_2]))
                if sim > 0.708:
                    trovato = True
                    sim_word.append(sim)
                    word2.pop(index_2)
                    break
            if not trovato:
                sim_word.append(sim)
    else:
        for index in range(min_2):
            trovato = False
            sim = 0
            min_1 = len(word1)
            for index_2 in range(min_1):
                sim = (config.wv.similarity(word2[index], word1[index_2]))
                if sim > 0.708:
                    trovato = True
                    sim_word.append(sim)
                    word1.pop(index_2)
                    break
            if not trovato:
                sim_word.append(sim)
    if sim_word == []:
        return None
    sim_word =  np.mean(sim_word)
    return sim_word

def is_in_list(word,list):
    for obj in list:
        if compare_two_words(word, obj) != None and compare_two_words(word, obj) > 0.600:
            return True
    return False

def list_to_yoloworld(list):
    list_objects = ""
    for object in list:
        if object != []:
            list_objects += (" ".join(object))
            list_objects += ", "
    return list_objects[:-2]


for relation in object_relations:

    print(relation)

    relation = relation.replace("(", "")
    relation_object_first = relation.split(")")[1].split(",")[0]
    relation_object_second = relation.split(")")[1].split(",")[2]
    word_1 = split_word(relation_object_first)
    word_2 = split_word(relation_object_second)
    print(word_1)
    print(word_2)
    if not is_in_list(word_1,relation_list):
        relation_list.append(word_1)
    if not is_in_list(word_2,relation_list):
        relation_list.append(word_2)

print(relation_list)
classes_to_detect =list_to_yoloworld(relation_list)
print(classes_to_detect)

config.yolow.set_class_name(classes_to_detect)


def find_index(label,list):
    for object_relations in list:
        if label in object_relations:
            return list.index(object_relations)
    return -1

relations_grounded = {}


for index_dict in range(len(data)):
    label = (data[index_dict]['label'])
    print(label)
    index_state = find_index(label,object_relations)
    print(object_relations[index_state])
    if index_state not in relations_grounded:
        relations_grounded[index_state] =  {"bbox":[], "label":[], "object_relations":[]}
    relations_grounded[index_state]['bbox'].append(data[index_dict]['bbox'])
    relations_grounded[index_state]['label'].append(data[index_dict]['label'])

print(object_relations)

import cv2

image = cv2.imread('diff1.jpg')

print(data)
image = cv2.imread('diff1.jpg')
colors = [[255,0,0],[0,255,0],[0,0,255],[255,255,0],[0,255,255],[255,0,255]]
ind_col = 0
for relation in object_relations:
    print(relation) 
    frist_object_in_detection = []
    second_object_in_detection = []
    relation_object_second = ""
    relation_object_first = "" 
    relation = relation.replace("(", "")
    relation_object_first = relation.split(")")[1].split(",")[0]
    relation_type = relation.split(")")[1].split(",")[1]
    relation_object_second = relation.split(")")[1].split(",")[2]
    for index_dict in range(len(data)):
        label = (data[index_dict]['label'])
        if label in relation_object_first:
            frist_object_in_detection.append(index_dict)
        if label in relation_object_second:
            second_object_in_detection.append(index_dict)
    if len(frist_object_in_detection) > 1 and len(second_object_in_detection) > 1:
        if "left" in relation_type:
            bbox_0 = data[frist_object_in_detection[0]]['bbox'] 
            bbox_1 = data[frist_object_in_detection[1]]['bbox']
            if bbox_1[0] > bbox_0[0]:
                index_first = frist_object_in_detection[1]
                index_second = second_object_in_detection[0]
            else:
                index_first = frist_object_in_detection[0]
                index_second = second_object_in_detection[1]
        
    if len(frist_object_in_detection) == 1:
        index_first = frist_object_in_detection[0]
    if len(second_object_in_detection) == 1:
        index_second = second_object_in_detection[0]
    
    print(index_first, relation_type , index_second)
    cv2.rectangle(image,(int(data[index_first]['bbox'][0]),int(data[index_first]['bbox'][1])),(int(data[index_first]['bbox'][2]),int(data[index_first]['bbox'][3])),colors[ind_col],3)
    cv2.putText(image,relation_object_first,(int(data[index_first]['bbox'][0]),int(data[index_first]['bbox'][1])),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
    cv2.rectangle(image,(int(data[index_second]['bbox'][0]),int(data[index_second]['bbox'][1])),(int(data[index_second]['bbox'][2]),int(data[index_second]['bbox'][3])),colors[ind_col],3)
    cv2.putText(image,relation_object_second,(int(data[index_second]['bbox'][0]),int(data[index_second]['bbox'][1])),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
    print("disegno la bb", relation_type, relation_object_first, relation_object_second)
    ind_col += 1
cv2.imwrite("image_23.jpg", image)
    