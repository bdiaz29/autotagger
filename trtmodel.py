import cv2
import numpy as np
import os
import csv
import copy
import argparse
from glob import glob

tags = None
include_characters = False

import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt

class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


class TrtTagger:
    def __init__(self, engine_path, max_batch_size=1):
        self.engine_path = engine_path
        self.dtype = np.float32
        self.logger = trt.Logger(trt.Logger.ERROR)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.load_engine(self.runtime, self.engine_path)
        shape = self.engine.get_binding_shape(0)
        self.batch_size = shape[0]
        self.height = shape[1]
        self.width = shape[2]
        self.max_batch_size = 1
        self.inputs, self.outputs, self.bindings, self.stream = self.allocate_buffers()
        self.context = self.engine.create_execution_context()


    @staticmethod
    def load_engine(trt_runtime, engine_path):
        trt.init_libnvinfer_plugins(None, "")
        with open(engine_path, 'rb') as f:
            engine_data = f.read()
        engine = trt_runtime.deserialize_cuda_engine(engine_data)
        return engine

    def allocate_buffers(self):
        inputs = []
        outputs = []
        bindings = []
        stream = cuda.Stream()

        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding)) * self.max_batch_size
            host_mem = cuda.pagelocked_empty(size, self.dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)

            bindings.append(int(device_mem))

            if self.engine.binding_is_input(binding):
                inputs.append(HostDeviceMem(host_mem, device_mem))
            else:
                outputs.append(HostDeviceMem(host_mem, device_mem))

        return inputs, outputs, bindings, stream

    def __call__(self, pre_processed_img_list):
        post_processed_img_list = []
        for img in pre_processed_img_list:
            h, w, c = img.shape
            mx = max(h, w)
            scale_x = self.width / mx
            scale_y = self.height / mx
            scale = min(scale_y, scale_y)
            canvas = np.ones((self.height, self.width, 3), dtype=np.uint8) * 255
            new_h = int(scale * h)
            new_w = int(scale * w)
            canvas[0:new_h, 0:new_w] = cv2.resize(img, (new_w, new_h))
            shift_y = int((self.height - new_h) / 2)
            shift_x = int((self.width - new_w) / 2)
            canvas = np.roll(canvas, shift_y, axis=0)
            canvas = np.roll(canvas, shift_x, axis=1)
            post_processed_img_list += [canvas]
        x = np.array(post_processed_img_list, dtype=self.dtype)
        # x = canvas.astype(self.dtype)
        np.copyto(self.inputs[0].host, x.ravel())
        for inp in self.inputs:
            cuda.memcpy_htod_async(inp.device, inp.host, self.stream)
        self.context.execute_async(batch_size=self.batch_size, bindings=self.bindings,
                                   stream_handle=self.stream.handle)
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out.host, out.device, self.stream)
        self.stream.synchronize()
        outtmp=[out.host.reshape(self.batch_size, -1) for out in self.outputs]

        return outtmp[0]

