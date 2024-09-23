from Handler import Handler

handler = Handler('LocalData', database_state=True)
import cv2
import argparse

parser = argparse.ArgumentParser(description='Facial Recognition Application')
parser.add_argument('--mode', '-m', type=int, default=0, help='0 - register new face into system, 1 - face recognition')
parser.add_argument('--vids', '-v', type=str, default='', help='if empty, auto search for webcam, else - a path to a folder contains videos you want to recognize')
parser.add_argument('--imgs', '-i', type=str, default='', help='if empty, auto search for webcam, else - a path to a folder contains images of new faces you want to register')

args = vars(parser.parse_args())

if args['imgs'] == '' and args['vids'] == '':
    handler.videoProcessing(use_camera=True, path='')
elif args['vids'] != '':
    handler.videoProcessing(use_camera=False, path=args['vids'])
else:
    handler.print_ID_results(args['imgs'])
    cv2.waitKey(0)
