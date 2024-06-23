from ultralytics import YOLO
from tracker import Tracker
import torch

class HumanTracker():
    
    def __init__(self):
        
        # initiate YOLOv8 for object detection & DeepSORT for object tracking
        self.model = YOLO("yolov8n.pt")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)
        print(f'YOLO using {device}.')
        self.tracker = Tracker()

        # id for drone following
        self.chosen_id = 0

    def detect_track(self,frame):

        detections = []
        tp = []
        ID = []
        h,w = frame.shape[:2]

        # only consider human with confidence score >0.5
        results = self.model(frame, conf=0.3, classes=[0])

        for result in results:
            det = []
            for r in result.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = r
                x1 = int(x1)
                x2 = int(x2)
                y1 = int(y1)
                y2 = int(y2)
                class_id = int(class_id)
                det.append([x1, y1, x2, y2, score])

            # detections from YOLO is passed to DeepSORT for object tracking
            self.tracker.update(frame, det)

            for track in self.tracker.tracks:
                bbox = track.bbox
                x1, y1, x2, y2 = bbox
                x1 = int(x1)
                x2 = int(x2)
                y1 = int(y1)
                y2 = int(y2)
                bbox = (x1, y1, x2-x1, y2-y1)
                detections.append(bbox)

                track_id = track.track_id
                ID.append(track_id)

            if len(detections) > 0:
                # construct target point, [midx,midy,position to track]
                # check area related to the full image
                area_frame = w*h
                area_det = detections[0][2] * detections[0][3]

                # multiply with 10, to keep the compatibility between detectors
                area_ratio = int((area_det/area_frame)*1000)
                
                # if chosen id is not detected in current frame, the first detection(smallest available id) will be target
                try:
                    tp_index = ID.index(self.chosen_id)
                except ValueError:
                    tp_index = 0
                tp = [detections[tp_index][0] + detections[tp_index][2]//2, detections[tp_index][1] + detections[tp_index][3]//3, area_ratio]

        print('tp: ', tp, ", detections: ", detections, 'ID: ', ID)
        return tp, detections, ID
            

