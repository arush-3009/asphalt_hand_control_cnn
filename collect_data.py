import cv2
import mediapipe as mp
import os
import time
import numpy as np

GESTURES = {
    '1': "open_hand",
    '2': "fist",
    '3': "v_sign",
    '4': "index_pointing",
    '5': "no_gesture"
}

IMG_SIZE = 224
CAPTURE_INTERVAL = 0.2
DATASET_PATH = "dataset/raw"
PADDING = 30 #Padding for bouding box aroung the hand in the image
