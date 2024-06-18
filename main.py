import cv2
import datetime
import imutils
import numpy as np
from centroidtracker import CentroidTracker
from concurrent.futures import ThreadPoolExecutor

# Load the model
protopath = "MobileNetSSD_deploy.prototxt"
modelpath = "MobileNetSSD_deploy.caffemodel"
detector = cv2.dnn.readNetFromCaffe(protopath, modelpath)

# Use GPU if available
detector.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
detector.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

tracker = CentroidTracker(maxDisappeared=80, maxDistance=90)

def non_max_suppression_fast(boxes, overlapThresh):
    if len(boxes) == 0:
        return []

    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    pick = []
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        overlap = (w * h) / area[idxs[:last]]
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))

    return boxes[pick].astype("int")

def adjust_lighting(frame):
    frame_yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
    frame_yuv[:, :, 0] = cv2.equalizeHist(frame_yuv[:, :, 0])
    frame = cv2.cvtColor(frame_yuv, cv2.COLOR_YUV2BGR)
    return frame

def process_frame(frame):
    frame = imutils.resize(frame, width=400)
    frame = adjust_lighting(frame)
    (H, W) = frame.shape[:2]
    line_position = W // 2
    cv2.line(frame, (line_position, 0), (line_position, H), (0, 0, 0), 2)

    blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)
    detector.setInput(blob)
    detections = detector.forward()

    rects = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.7:
            idx = int(detections[0, 0, i, 1])
            if CLASSES[idx] == "person":
                box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                rects.append(box.astype("int"))

    rects = non_max_suppression_fast(np.array(rects), 0.5)
    return frame, rects

def main():
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    fps_start_time = datetime.datetime.now()
    total_frames = 0
    entry_count = 0
    exit_count = 0
    object_state = {}
    line_position = 200
    paused = False

    with ThreadPoolExecutor(max_workers=2) as executor:
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    break

                future = executor.submit(process_frame, frame)
                frame, rects = future.result()
                total_frames += 1

                objects = tracker.update(rects)

                for (objectId, bbox) in objects.items():
                    x1, y1, x2, y2 = bbox
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    text = "ID: {}".format(objectId)
                    cv2.putText(frame, text, (x1, y1 - 5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)

                    centroid = ((x1 + x2) // 2, (y1 + y2) // 2)
                    if objectId not in object_state:
                        object_state[objectId] = {'previous_centroid': centroid, 'counted_entry': False, 'counted_exit': False}

                    prev_centroid = object_state[objectId]['previous_centroid']
                    object_state[objectId]['previous_centroid'] = centroid

                    if prev_centroid[0] > line_position and centroid[0] <= line_position:
                        if not object_state[objectId]['counted_entry']:
                            entry_count += 1
                            object_state[objectId]['counted_entry'] = True

                    if prev_centroid[0] < line_position and centroid[0] >= line_position:
                        if object_state[objectId]['counted_entry'] and not object_state[objectId]['counted_exit']:
                            exit_count += 1
                            object_state[objectId]['counted_exit'] = True

                headcount = entry_count - exit_count

                cv2.putText(frame, "Entry: {}".format(entry_count), (5, 120), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)
                cv2.putText(frame, "Exit: {}".format(exit_count), (5, 150), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)
                cv2.putText(frame, "Headcount: {}".format(headcount), (5, 180), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)

                fps_end_time = datetime.datetime.now()
                time_diff = fps_end_time - fps_start_time
                fps = (total_frames / time_diff.total_seconds()) if time_diff.total_seconds() > 0 else 0.0
                cv2.putText(frame, "FPS: {:.2f}".format(fps), (5, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)

                lpc_count = len(objects)
                opc_count = len(object_state)
                cv2.putText(frame, "LPC: {}".format(lpc_count), (5, 60), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)
                cv2.putText(frame, "OPC: {}".format(opc_count), (5, 90), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)

                current_ids = set(objects.keys())
                old_ids = set(object_state.keys())
                for old_id in old_ids - current_ids:
                    del object_state[old_id]

            cv2.imshow("Application", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('w'):
                paused = not paused
            elif key == ord('r'):
                # Reset counts and object state
                entry_count = 0
                exit_count = 0
                headcount = 0
                object_state = {}
                cv2.putText(frame, "Data Reset", (5, 210), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 1)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
