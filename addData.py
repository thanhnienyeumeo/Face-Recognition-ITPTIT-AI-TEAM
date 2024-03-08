import argparse
import os


import argparse

parser = argparse.ArgumentParser(description='Facial Recognition Application')
parser.add_argument('--mode', '-m', type=int, default=0, help='0 - register new face into system, 1 - face recognition')
parser.add_argument('--imgs', '-i', type=str, default='', help='if empty, auto search for webcam, else - a path to a folder contains images of new face')
parser.add_argument('--database', '-dp', type=str, default='database', help='Path to parent \'database\' path, contain sub-folders in which contain face images')
parser.add_argument('--backend', '-dbe', type=str, default='retina', help='Backend for face detection')
args = vars(parser.parse_args())

print(args['mode'])

if args['mode'] == 1:
    