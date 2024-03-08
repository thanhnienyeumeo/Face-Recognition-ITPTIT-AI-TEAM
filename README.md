# FACE RECOGNITION SYSTEM (FOR BOTH WINDOWS AND LINUX) 

## Prepare for environment
INSTALL CUDA AND CUDNN (my version: cuda 11.2, cudnn 8.2.2.26)
Using virtual environment if you can

NOTE: IF YOU ARE USING WINDOWS, download Visual Studio here
https://visualstudio.microsoft.com/visual-cpp-build-tools/
```sh
git clone https://github.com/thanhnienyeumeo/Face-Recognition-ITPTIT-AI-TEAM.git
pip install -r requirements.txt
```

if you can't run code on gpu, select one of these options:

Using CPU (very slow for real-time recognition):
```sh
pip uninstall onnxruntime-gpu
pip install onnxruntime
```
OR

Check onnxruntime-gpu version that compatible with your cuda and cudnn version here https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html



## TRAINING
Developing.....

## TESTING
### Run video.py:

Download Dataset here: https://www.kaggle.com/datasets/stoicstatic/face-recognition-dataset/versions/4

My embeddings was created by using this dataset

Change path and simply run code
### USING
Run main.py

Step 1: Create a folder named LocalData

Step 2: Create a folder named your face's name

Step 3: Add a image of that face to the folder in step 2



Step 4: Run main.py (read all agruments if you want to use image or videos instead of camera, or not, just run with no agrument)

### FOR Developer want to understand code:
Full process code:

Run .ipynb file first
