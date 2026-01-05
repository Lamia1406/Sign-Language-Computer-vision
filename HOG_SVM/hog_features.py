import cv2
from skimage.feature import hog

def extract_hog_features(img_rgb, hog_params, size=(64, 64)):
    img_resized = cv2.resize(img_rgb, size)
    img_gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
    return hog(img_gray, **hog_params)
