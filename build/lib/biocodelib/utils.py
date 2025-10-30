import cv2

def load_image(path):
    return cv2.imread(path)

def save_image(image, path):
    cv2.imwrite(path, image)