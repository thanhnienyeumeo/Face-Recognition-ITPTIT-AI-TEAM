from ArcFace import ArcFaceONNX
from SCRFD import SCRFD
import glob
import os
import cv2
from functions import generate_embs, plot_one_box, Area, resize_img
import numpy as np
import time
from sklearn.neighbors import NearestNeighbors
import insightface



class Handler():
    def __init__(self, database_path, algorithm = 'knn', model_name = 'buffalo_l') -> None:
        self.assets_dir = os.path.expanduser('~/.insightface/models/buffalo_l')
        self.arcface = insightface.app.FaceAnalysis(model_name)
        self.arcface = self.arc_model()
        self.detector = self.arc_detector()
        self.embeeding_path = 'Embeeding'
        # if backend == 'retina':
        #     self.detector = RetinaDetector()
        # elif backend == 'opencv':
        #     self.detector = CascadeDetector()
        # self.retina = RetinaDetector()
        # self.haar_cascade = CascadeDetector()
        self.mean_face_database = []
        self.face_database = []
        self.image_size = (112, 112) # for arcface
        # True to use mean-feature verification, False for single-feature verification (take more time)
        self.verify_mode = False   
        self.database_state = False
        self.database_path = database_path
        if algorithm == 'knn':
            self.algorithm = NearestNeighbors(n_neighbors=3, metric="cosine")
        else:
            self.algorithm = None
        # initialize database automatically
        self.init_identity_database(database_path)
        
    
    def arc_model(self):
        rec = ArcFaceONNX(os.path.join(self.assets_dir, 'w600k_r50.onnx'))
        rec.prepare(0)
        return rec

    def arc_detector(self):
        detector = SCRFD(os.path.join(self.assets_dir, 'det_10g.onnx'))
        detector.prepare(0)
        return detector

    def init_identity_database(self, parent_folder_path='LocalData'):
        # check if database needs update or not
        if self.database_state == True:
            return
        
        # reset database
        self.mean_face_database = []
        self.face_database = []

        files = os.listdir(parent_folder_path)
        all_embs, all_labels = [], []
        
        for identity_folder in files:
            
            identity_path = os.path.join(parent_folder_path, identity_folder)

            all_image = os.listdir(identity_path)
            
            embs, labels = generate_embs(all_image, self.arcface, self.detector, identity_folder, identity_path)
            
            if len(embs) != 0:
                all_embs.extend(embs)
                all_labels.extend(labels)
            else: print('Error in ')
            print(identity_folder)
       
        np.save(self.embeeding_path + '/embs.npy', all_embs)
        np.save(self.embeeding_path + '/labels.npy', all_labels)
        self.algorithm.fit(X = all_embs)
        # print(self.face_database)
        self.database_state = True
    
    def print_ID_results(self, img, verbose: bool = False):
    #print(img)
        
        app = self.arcface
        labels = np.load(self.embeeding_path + '/labels.npy')
        #print(img)
        t1 = time.time()
        if type(img) == str:
            img = cv2.imread(img)
        if img.shape[0] > 2000 or img.shape[1] > 2000:
            img = resize_img(img, 50)
        boxs, kpss = self.detector.autodetect(img)
        for i in range(len(kpss)):
            img_emb = app.get(img, kpss[i])
      
      # get pred from KNN
      
            dists, inds = self.algorithm.kneighbors(X=img_emb.reshape(1,-1), n_neighbors=3, return_distance=True)
      
      
      # get labels of the neighbours
            pred_labels = [labels[i] for i in inds[0]]
      
      # check if any dist is greater than 0.5, and if so, print the results
            no_of_matching_faces = np.sum([1 if d <=0.6 else 0 for d in dists[0]])
            if no_of_matching_faces > 0:
                verbose = True
                print(pred_labels[0], type(pred_labels[0]))
                plot_one_box(boxs[i][:4], img, label = str(pred_labels[0]))
        #   print("Matching face(s) found in database! ")
        #   for label, dist in zip(pred_labels, dists[0]):
        #     print(f"Nearest neighbours found in the database have labels {label} and is at a distance of {dist}")
            else:
                plot_one_box(boxs[i][:4], img, label = 'unknown')
        cv2.imshow('',img)
        t2 = time.time()
        
    #print(t2-t1)
    
    """
        This function take in an image, process any face detected within and return 
        a frame with face's bounding boxes, name and confidence score
    """
    
    def videoProcessing(self, use_camera, path = None):
    # Mở video để đọc
        if use_camera:
            video_capture = cv2.VideoCapture(0)
            # video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
            # video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 2880)
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
        window_closed = False
        # cv2.namedWindow("Video", cv2.WND_PROP_FULLSCREEN)
        # cv2.setWindowProperty("Video", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        print(2.5)
        while True:
            # Đọc từng frame của video
            ret, frame = video_capture.read()
            
            # Kiểm tra xem frame có đọc thành công hay không
            if not ret:
                break
            
            #frame = cv2.flip(frame, 0)

            cut_frame = (cut_frame + 1) % ratio_cut_frame
            if cut_frame: continue
            # Xử lý từng frame
            t1 = time.time()
            
            self.print_ID_results(frame, verbose = True)
            t2 = time.time()
            max_delay = max(max_delay, t2 - t1)
            #print(max_delay)
            key = cv2.waitKey(5)
            if key == 27:
                break
            if cv2.getWindowProperty("Original Video", cv2.WND_PROP_VISIBLE) < 1:
                window_closed = True
            # if window_closed:
            #    break
            
        # Giải phóng video và đóng cửa sổ hiển thị
        video_capture.release()
        cv2.destroyAllWindows()
        print(max_delay)

    def register_identity(self, img=cv2.imread('', cv2.IMREAD_COLOR), identity=''):
        # detect face from image
        faces, landms = self.detector.detect(img)
        # initialize parameters
        msg = ''
        # check if only one face detected in frame
        if len(faces) > 1:
            msg = 'More than one face detected!'
            print(msg)
            # print(type([]), type(img))
            return [], img
        else:
            # this for loop is only fool-proof, program logic will only add ONE person in ONE frame at a time.
            for idx in range(len(faces)):
                # getting face bounding box coordinates
                x1, y1, x2, y2 = faces[idx][0], faces[idx][1], faces[idx][2], faces[idx][3]
                # cut face from image
                face_frame = img[y1:y2, x1:x2]
                # resize image to desired size
                face_frame = cv2.resize(face_frame, (112, 112), interpolation=cv2.INTER_AREA)
                # get 'this' face landmarks
                landmk = landms[idx]
                # get input blob
                blob = self.arcface.get_image(face_frame, landmk)
                # get face feature
                feat = self.arcface.forward(blob)

                cv2.rectangle(img, (x1, y1), (x2, y2), (0,0,255), 2)
                cv2.putText(img, str(faces[idx][4]), (x1, y1 + 10), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,0,0))
                # print(type(feat), type(img))
                self.database_state = False
                return feat, img
        return [], img
    def cal_and_app_feature(self, features, identity_name):
        if len(features) == 0:
            return
        for f in features:
            self.face_database.append((identity_name, f))
        mean_f = np.zeros((1, 1024))
        mean_f = np.mean(features, axis=0)
        self.mean_face_database.append((identity_name, mean_f))