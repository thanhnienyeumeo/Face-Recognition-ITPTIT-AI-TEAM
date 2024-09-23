import argparse
import cv2
import sys
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image

def plot_one_box(x, img, color=None, label=None, line_thickness=3):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)



parser = argparse.ArgumentParser(description='insightface gender-age test')
# general
parser.add_argument('--ctx', default=0, type=int, help='ctx id, <0 means using cpu')
args = parser.parse_args()
model_name = 'buffalo_l'
app = FaceAnalysis(model_name, allowed_modules=['detection', 'genderage'])
app.prepare(ctx_id=args.ctx, det_size=(640,640))

img_path = r'LocalData\ThanhDua\z5231364534036_ad51359c20d1ba6ef8c849b2c6e9e1fd.jpg'
#show img
img = cv2.imread(img_path)
faces = app.get(img)

for face in faces:
    # print(face.bbox)
    # print(face.sex, face.age)
    label ='Female' if face.sex == 'F' else 'Male'
    label += ', ' + str(face.age)
    #print sex and age in image
    plot_one_box(face.bbox, img, label = label)
cv2.imshow('image', img)
cv2.waitKey(0)