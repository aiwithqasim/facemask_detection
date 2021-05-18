# import the required packages
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import cv2


def detect_and_predict_mask(frame, faceNet, maskNet):
    # get the face points for detection
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
                                 (104.0, 177.0, 123.0))
    faceNet.setInput(blob)
    detections = faceNet.forward()
    print(detections.shape)

    # Initialization for faces, location & prediction
    faces = []
    local_points = []
    prediction = []

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence for detection
        confidence = detections[0, 0, i, 2]

        # validating confidence level
        if confidence > 0.5:
            # measuring the coordinates of the bounding box
            bounding_box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (start_x, start_y, end_x, end_y) = bounding_box.astype("int")

            # validating the bounding boxes
            (start_x, start_y) = (max(0, start_x), max(0, start_y))
            (end_x, end_y) = (min(w - 1, end_x), min(h - 1, end_y))

            # extract the face ROI, color transformation & resizing
            face = frame[start_y:end_y, start_x:end_x]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)

            # face and bounding boxes to their respective list
            faces.append(face)
            local_points.append((start_x, start_y, end_x, end_y))

    # searching for one face in frame minimumly
    if len(faces) > 0:
        # for faster inference we'll make batch predictions on *all*
        # faces at the same time rather than one-by-one predictions
        # in the above `for` loop
        faces = np.array(faces, dtype="float32")
        prediction = maskNet.predict(faces, batch_size=32)

    # return a tuple of the face locations
    return local_points, prediction


# load our serialized face detector model from disk
prototxtPath = r"face_detector\deploy.prototxt"
weightsPath = r"face_detector\res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# loading the  model
maskNet = load_model("mask_detector.model")

# starting the video stream
print("starting video stream...")
vs = VideoStream(src=0).start()

# infinite loop
while True:
    # grab threaded video stream, resize it.
    frame = vs.read()
    frame = imutils.resize(frame, width=400)

    # detect faces and determine face mask
    (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

    # detected face locations and their corresponding locations
    for (box, pred) in zip(locs, preds):
        (startX, startY, endX, endY) = box
        (mask, withoutMask) = pred

        # detecting the class label and color
        label = "Mask" if mask > withoutMask else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

        # adding the probability
        label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

        # display the label and bounding box
        cv2.putText(frame, label, (startX, startY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

    # show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # setting up breaking condition with 'q'
    if key == ord("q"):
        break

# destroying window
cv2.destroyAllWindows()
vs.stop()
