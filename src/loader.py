import os
import time
from models import YOLOW, VitSam

import spacy
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import gensim.downloader as api

class Loader:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(Loader, cls).__new__(cls)
            cls._instance.initialize(*args, **kwargs)
        return cls._instance

    def initialize(self):
        self._CONFIG = os.path.join(os.getcwd(),"config/")
        self._IMAGES = os.path.join(os.getcwd(),"images/")
        self._AGENTS = os.path.join(os.getcwd(),"config/agents/")

        self._YOLOW_PATH = self._CONFIG + "yolow/"
        self._ENCODER_PATH = self._CONFIG + "efficientvitsam/l2_encoder.onnx"
        self._DECODER_PATH = self._CONFIG + "efficientvitsam/l2_decoder.onnx"

        self._IMAGE_DIR = self._IMAGES + "test_order/"

        self._DUMP = self._CONFIG + "dump_order/"

        self._yolow_model = YOLOW(self._YOLOW_PATH)
        self._vit_sam_model = VitSam(self._ENCODER_PATH, self._DECODER_PATH) 

        self._nlp = spacy.load("en_core_web_sm") 
        self._wv = api.load('word2vec-google-news-300') 

    @property
    def nlp(self):
        return self._nlp
    
    @nlp.setter
    def nlp(self, value):
        self._nlp = value

    @property
    def wv(self):
        return self._wv
    
    @wv.setter
    def wv(self, value):
        self._wv = value

    @property
    def yolow_model(self):
        return self._yolow_model
    
    @yolow_model.setter
    def yolow_model(self, value):
        self._yolow_model = value

    @property
    def vit_sam_model(self):
        return self._vit_sam_model
    
    @vit_sam_model.setter
    def vit_sam_model(self, value):
        self._vit_sam_model = value
        
    @property
    def CONFIG(self):
        return self._CONFIG
    
    @CONFIG.setter
    def CONFIG(self, value):
        self._CONFIG = value
    
    @property
    def IMAGES(self):
        return self._IMAGES
    
    @IMAGES.setter
    def IMAGES(self, value):
        self._IMAGES = value
    
    @property
    def YOLOW_PATH(self):
        return self._YOLOW_PATH
    
    @YOLOW_PATH.setter
    def YOLOW_PATH(self, value):
        self._YOLOW_PATH = value

    @property
    def ENCODER_PATH(self):
        return self._ENCODER_PATH
    
    @ENCODER_PATH.setter
    def ENCODER_PATH(self, value):
        self._ENCODER_PATH = value
    
    @property
    def DECODER_PATH(self):
        return self._DECODER_PATH
    
    @DECODER_PATH.setter
    def DECODER_PATH(self, value):
        self._DECODER_PATH = value
    
    @property
    def IMAGE_DIR(self):
        return self._IMAGE_DIR
    
    @IMAGE_DIR.setter
    def IMAGE_DIR(self, value):
        self._IMAGE_DIR = value
    
    @property
    def DUMP(self):
        return self._DUMP
    
    @DUMP.setter
    def DUMP(self, value):
        self._DUMP = value
    
    @property
    def AGENTS(self):
        return self._AGENTS
    
    @AGENTS.setter
    def AGENTS(self, value):
        self._AGENTS = value
    