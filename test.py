import face_align
import onnx
import onnxruntime
import os
import pickle
import numpy as np

from tqdm import tqdm
import insightface
from sklearn.neighbors import NearestNeighbors
from SCRFD import SCRFD
from ArcFace import ArcFaceONNX
import cv2
#app = FaceAnalysis(name="antelope")
path = 'testData\z5231335250816_89b0739b188ee9912dcd3a7519a89dd2.jpg'
pathVuaHon = 'testData\z5231334708281_c512eecab5c03095eb72d7be055c6308.jpg'
img = cv2.imread(pathVuaHon)
print(img.shape)
cv2.imshow('image', img)
cv2.waitKey(0)
# scale_percent = 20 # percent of original size
# width = int(img.shape[1] * scale_percent / 100)
# height = int(img.shape[0] * scale_percent / 100)
# dim = (width, height)
  
# # resize image
# resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
# cv2.imshow('', resized)
# cv2.waitKey(0)