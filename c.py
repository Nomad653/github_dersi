import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    cv2.imshow('frame',frame)

