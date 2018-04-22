import atexit
import numpy as np
from os import environ

ENV_KEY = 'TENSOR_SAVER_PATH'

class Saver:

    def __init__(self, path):
        self.path = path
        self.objs = []

    def add(self, obj):
        if len(obj.shape) == 0:
            obj = np.array([obj])
        self.objs.append(obj)

    def save(self):
        dobjs= {}
        for i in range(len(self.objs)):
            name = 'obj_' + str(i).zfill(6)
            dobjs[name] = self.objs[i]
        np.savez(self.path, **dobjs)

_gbl_saver = Saver(environ[ENV_KEY] if ENV_KEY in environ else './debug.npz')

def add(obj):
    _gbl_saver.add(obj)

def set_out_path(path):
    _gbl_saver.path = path

def save():
    _gbl_saver.save()

def _on_exit():
    if len(_gbl_saver.objs) != 0:
       _gbl_saver.save()

atexit.register(_on_exit)
