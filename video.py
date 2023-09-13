import face_align
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
import time
from ArcFace import ArcFaceONNX
import matplotlib.pyplot as plt


model_name = 'buffalo_l'
app = insightface.app.FaceAnalysis(model_name)
assets_dir = os.path.expanduser('~/.insightface/models/buffalo_l')

def arc_detector():
  detector = SCRFD(os.path.join(assets_dir, 'det_10g.onnx'))
  detector.prepare(0)
  return detector

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

def Area(rect):
  return (rect[2] - rect[0]) * (rect[3] - rect[1])

def arc_model():
  rec = ArcFaceONNX(os.path.join(assets_dir, 'w600k_r50.onnx'))
  rec.prepare(0)
  return rec

# some time later...
#load the model from disk
def load_KNN():
    nn = None
    path = 'Data/Data'
    with open(path + '/faceID_model.pkl', 'rb') as file:
        nn = pickle.load(file)
    print(nn)
    return nn
#load the embeeding
def load_embs():
    path = 'Data/Data'
    embs = np.load(path + '/embs.npy')
    labels = np.load(path + '/labels.npy')
    probe_embs = np.load(path + '/probe_embs.npy')
    probe_labels = np.load(path + '/probe_labels.npy')
    return embs, labels, probe_embs, probe_labels


import time
def print_ID_results(img, evaluation_labels: np.ndarray, detector, app, nn, verbose: bool = False):
    #print(img)
    t1 = time.time()
    boxs, kpss = detector.autodetect(img)
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
        #   print("Matching face(s) found in database! ")
        #   for label, dist in zip(pred_labels, dists[0]):
        #     print(f"Nearest neighbours found in the database have labels {label} and is at a distance of {dist}")
      else:
          plot_one_box(boxs[i][:4], img, label = 'unknown')
    cv2.imshow('',img)
    t2 = time.time()
    print(t2-t1)


#app = FaceAnalysis(name="antelope")


def videoProcessing(use_camera, path, detector, app, labels, nn):
    # Mở video để đọc
    if use_camera:
       video_capture = cv2.VideoCapture(0)
    else:
       video_path = path
       video_capture = cv2.VideoCapture(path)
    
    
    cut_frame = 0
    ratio_cut_frame = 3
    # Kiểm tra xem video có mở thành công hay không
    if not video_capture.isOpened():
        print("Không thể mở video!")
        return
    max_delay = 0
    while True:
        # Đọc từng frame của video
        ret, frame = video_capture.read()
        
        # Kiểm tra xem frame có đọc thành công hay không
        if not ret:
            break
        if cut_frame: continue
        cut_frame = (cut_frame + 1) % ratio_cut_frame
        # Xử lý từng frame
        t1 = time.time()
        print_ID_results(frame, labels, detector=detector, app=app, nn=nn, verbose = True)
        t2 = time.time()
        max_delay = max(max_delay, t2 - t1)
        #print(max_delay)
        if cv2.waitKey(5) == ord(str('q')):
            break
    # Giải phóng video và đóng cửa sổ hiển thị
    video_capture.release()
    cv2.destroyAllWindows()
    print(max_delay)

detector = arc_detector()
app = arc_model()
#img = cv2.imread('DSC_3459.jpg')
#a = detector.autodetect(img)
embs, labels, probe_embs, probe_labels = load_embs()
# Train KNN classifier
nn = NearestNeighbors(n_neighbors=3, metric="cosine")
nn.fit(X=embs)
videoProcessing(True, 'img.MOV', detector, app, labels, nn) #Set use_camera = True to use camera, False to use video
