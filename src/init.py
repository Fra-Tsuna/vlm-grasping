import os
import time
from models import YOLOW, VitSam

class Loader:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(Loader, cls).__new__(cls)
            cls._instance.initialize(*args, **kwargs)
        return cls._instance

    def initialize(self):
        self.CONFIG = os.path.join(os.getcwd(),"config/")
        self.IMAGES = os.path.join(os.getcwd(),"images/")
        time.sleep(10)

        self.YOLOW_PATH = self.CONFIG + "yolow/"
        #YOLOW_PATH = CONFIG + "yolow/sort.onnx"
        self.ENCODER_PATH = self.CONFIG + "efficientvitsam/l2_encoder.onnx"
        self.DECODER_PATH = self.CONFIG + "efficientvitsam/l2_decoder.onnx"

        self.IMAGE_DIR = self.IMAGES + "test_order/"

        DUMP = self.CONFIG + "dump_order/"
        self.yolow_model = YOLOW(self.YOLOW_PATH)

    
    @property
    def yolow_model(self):
        return self._yolow_model

    @yolow_model.setter
    def yolow_model(self, value):
        self._yolow_model = value


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
    
