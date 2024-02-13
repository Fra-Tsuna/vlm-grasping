from typing import Any


class CircularBuffer:
    def __init__(self, size):
        self.size = size
        self.data = [None] * size
        self.index = 0

    def append(self, item):
        self.data[self.index] = item
        self.index = (self.index + 1) % self.size

    def get(self, i):
        return self.data[i % self.size]

    def __len__(self):
        return self.size
    

class View:
    def __init__(self, image, description):
        self.image = image
        self.description = description