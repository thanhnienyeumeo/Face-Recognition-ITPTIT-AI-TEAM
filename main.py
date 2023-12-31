# -*- coding: utf-8 -*-
"""faceRecognition.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1HmJuoQzbRenxNGlMbuVcL-decKIDIDFm
"""



import face_align as face_align
import onnx
import onnxruntime
import os
import pickle
import numpy as np
from PIL import Image
from typing import List
from tqdm import tqdm
import insightface
from sklearn.neighbors import NearestNeighbors
import cv2
from SCRFD import SCRFD
from ArcFace import ArcFaceONNX
import time

"""<h1> install buffalo_l model </h1>
<a>'det_10g.onnx': detection </a> <br/>
<c>'w600k_r50.onnx': </c> 
"""

#app = FaceAnalysis(name="antelope")
model_name = 'buffalo_l'
app = insightface.app.FaceAnalysis(model_name)
assets_dir = os.path.expanduser('~/.insightface/models/buffalo_l')
print(os.listdir(assets_dir))

print(onnxruntime.get_device())

def arc_detector():
  detector = SCRFD(os.path.join(assets_dir, 'det_10g.onnx'))
  detector.prepare(0)
  return detector

detector = arc_detector()

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

"""Test for a image have two more face

"""


def Area(rect):
  return (rect[2] - rect[0]) * (rect[3] - rect[1])




def arc_model():
  rec = ArcFaceONNX(os.path.join(assets_dir, 'w600k_r50.onnx'))
  rec.prepare(0)
  return rec

app = arc_model()

def generate_embs(img_fpaths: List[str], app, detector, label, path):
    embs_set = list()
    embs_label = list()

    for img_fpath in img_fpaths:  
       #read image
        img = cv2.imread(os.path.join(path, img_fpath))
        boxs, kpss = detector.autodetect(img)
        if len(kpss) == 0:
          print(f'{img_fpath} of number {label} Error. No face spotted. Skip')
          continue
      #  # generate Insightface embedding
        if len(kpss) > 1: #choose the largest one
          kpss = [x for _,x in sorted(zip(boxs, kpss), key = lambda e: -Area(e[0]))]
        try:
          emb_res = app.get(img, kpss[0])         
          embs_set.append(emb_res)          
          embs_label.append(label)
        except:
          print(f'\nno embedding found for this image: {img_fpath} of number {label}')         

    return embs_set, embs_label

"""<h1> test function, don't run this c
ell <h1>
"""

def generate_embs_without_detector(img_fpaths: List[str], app, label, path):
    embs_set = list()
    embs_label = list()

    for img_fpath in img_fpaths:  
       #read image
        img = cv2.imread(os.path.join(path, img_fpath))
        try:
          emb_res = app.get(img, kpss[0])         
          embs_set.append(emb_res)          
          embs_label.append(label)
        except:
          print(f'\nno embedding found for this image: {img_fpath} of number {label}')         

    return embs_set, embs_label



extracted_dir = "/content/drive/MyDrive/AI/Face_Recognition/Extracted Faces"
files_extracted = os.listdir(extracted_dir)
files_extracted.sort(key = lambda x: int(x))

import time

for i in files_extracted:
  path = input_dir + '/' + i
  if len(os.listdir(path)) < 5: continue
  total += len(os.listdir(path))
  #
  path = input_dir + '/' + i
  allFile = os.listdir(path)
  t1 = time.time()
  emb, label = generate_embs_without_detector(allFile, app, str(i) ,path)
  tt = time.time() - t1
  print(f"Finished at {i}. Time: {tt} for {len(os.listdir(path))} images")

"""<h1> END TESTING. THE REST OF CODE IS NORMAL <h1>"""

input_dir = "/content/drive/MyDrive/AI/Face Dataset" #input
files = os.listdir(input_dir)
files.sort(key = lambda x: int(x))

total = 0
for i in files:
  path = input_dir + '/' + i
  if len(os.listdir(path)) >= 5: total+=1
print(total) #316

from sklearn.model_selection import train_test_split

def generate_train_and_test(path_dir):
  train, test = train_test_split(path_dir)
  return train, test

embs = list()
labels = list()
probe_embs = list()
probe_labels = list()
start = time.time()
total = 0
for i in files:
  path = input_dir + '/' + i
  if len(os.listdir(path)) < 5: continue
  total += len(os.listdir(path))
  #
  t1 = time.time()
  path = input_dir + '/' + i
  allFile = os.listdir(path)

  eval, probe = generate_train_and_test(allFile)
  emb, label = generate_embs(eval, app, detector, str(i) ,path)
  embs.extend(emb)
  labels.extend(label)
  probe_emb, probe_label = generate_embs(probe, app, detector, str(i), path)
  probe_embs.extend(probe_emb)
  probe_labels.extend(probe_label)
  tt = time.time() - t1
  print(f"Finished at {i}. Time: {tt} for {len(os.listdir(path))} images")

end = time.time()
print(f"Embeeding {total} image: {end-start} seconds")

print(len(embs), len(probe_embs))
assert len(embs) == len(labels)
assert len(probe_embs) == len(probe_labels)

path = '/content/drive/MyDrive/AI/Face_Recognition'
np.save(path + '/embs.npy', embs)
np.save(path + '/labels.npy', labels)
np.save(path + '/probe_embs.npy', probe_embs)
np.save(path + '/probe_labels.npy', probe_labels)

# Train KNN classifier
nn = NearestNeighbors(n_neighbors=3, metric="cosine")
nn.fit(X=embs)


# save the model to disk
filename = '/content/drive/MyDrive/AI/Face_Recognition/faceID_model.pkl'
with open(filename, 'wb') as file:
    pickle.dump(nn, file)
    
# some time later...

#load the model from disk
path = '/content/drive/MyDrive/AI/Face_Recognition'
with open(path + '/faceID_model.pkl', 'rb') as file:
    nn = pickle.load(file)
#load the embeeding
embs = np.load(path + '/embs.npy')
labels = np.load(path + '/labels.npy')
probe_embs = np.load(path + '/probe_embs.npy')
probe_labels = np.load(path + '/probe_labels.npy')

dists, inds = nn.kneighbors(X=probe_embs, n_neighbors=2, return_distance=True)
pred_labels = [labels[i] for i in inds[1]] 
pred_labels

"""Evaluating metrics - **p_at_k**"""

# p@k
p_at_k = np.zeros(len(probe_embs))
for i in range(len(probe_embs)):
    true_label = probe_labels[i]
    pred_neighbr_idx = inds[i]
    
    pred_labels = [labels[id] for id in pred_neighbr_idx]
    pred_is_labels = [1 if label == true_label else 0 for label in pred_labels]
    
    p_at_k[i] = np.mean(pred_is_labels)
    
p_at_k.mean()

import time
def print_ID_results(img_fpath: str, evaluation_labels: np.ndarray, verbose: bool = False):

    img = cv2.imread(img_fpath)
    boxs, kpss = detector.autodetect(img)
    if len(kpss) == 0:
      print("Can't detect any face in this image")
      return
    for i in range(len(kpss)):
      img_emb = app.get(img, kpss[i])
      
      # get pred from KNN
      dists, inds = nn.kneighbors(X=img_emb.reshape(1,-1), n_neighbors=3, return_distance=True)
      
      # get labels of the neighbours
      pred_labels = [evaluation_labels[i] for i in inds[0]]
      
      # check if any dist is greater than 0.5, and if so, print the results
      no_of_matching_faces = np.sum([1 if d <=0.3 else 0 for d in dists[0]])
      if no_of_matching_faces > 0:
          verbose = True
          plot_one_box(boxs[i][:4], img, label = str(pred_labels[0]))
          print("Matching face(s) found in database! ")
          for label, dist in zip(pred_labels, dists[0]):
            print(f"Nearest neighbours found in the database have labels {label} and is at a distance of {dist}")
    # print labels and corresponding distances
    if not verbose: print('No matching face(s) found on database!')
    cv2_imshow(img)

print_ID_results("/content/drive/MyDrive/AI/Face Dataset/1454/2.jpg", labels, verbose=True)

print_ID_results("/content/drive/MyDrive/AI/Face Dataset/00/LPHoang.jpg", labels, verbose=True)

print_ID_results("/content/drive/MyDrive/AI/Face Dataset/00/NTHoang.jpg", labels, verbose=True)

print_ID_results("/content/drive/MyDrive/AI/Face Dataset/721/3.jpg", labels, verbose=True)

print_ID_results("/content/drive/MyDrive/AI/test/subject15_test.jpg", evaluation_labels, verbose=True)