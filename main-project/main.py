import cv2
import imutils
from imutils.object_detection import non_max_suppression
import numpy as np
import argparse

# Initialize the argument parser
parser = argparse.ArgumentParser(description='Pedestrian Detection using OpenCV')

# Add argument for the video file path
parser.add_argument('-v', '--video', type=str, required=True, help='Path to the video file')

# Parse the arguments
args = parser.parse_args()

# Use the provided video file path
cap = cv2.VideoCapture(args.video)

# Ensure the video file is opened correctly
if not cap.isOpened():
    print(f"Error: Could not open video file at {args.video}")
    exit()

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

while cap.isOpened():
    ret, image = cap.read()
    if ret:
        image = imutils.resize(image, width=min(400, image.shape[1]))
        (regions, _) = hog.detectMultiScale(image, 
                                            winStride=(4, 4), 
                                            padding=(4, 4), 
                                            scale=1.05)

        regions = np.array([[x, y, x + w, y + h] for (x, y, w, h) in regions])
        pick = non_max_suppression(regions, probs=None, overlapThresh=0.65)
        for (xA, yA, xB, yB) in pick:
            cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 2)

        cv2.imshow("Image", image) 
        if cv2.waitKey(25) & 0xFF == ord('q'): 
            break
    else: 
        break

cap.release()
cv2.destroyAllWindows()
