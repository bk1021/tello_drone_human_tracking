import cv2
import threading
import numpy as np
from . import safethread
from . import kalman
from . import humantracking


class FollowObject():
    """
    Commands the Drone to follow an object, defined by the loaded model.
    Horizontal / vertical / FW/BackW / yaw are controlled, using Kalman filters.
    """

    def __init__(self, tello) -> None:
        
        self.model = humantracking.HumanTracker()

        self.tello = tello

        # Kalman estimators
        self.kf = kalman.clKalman()
        self.kfarea= kalman.clKalman()

        # ticker for timebase
        self.ticker = threading.Event()

        # init 
        self.track = False
        self.img = None
        self.det = None
        self.tp = None
        self.ID = None
        # drone following
        self.follow = False
        # Is the drone flying?
        self.flying = False

        # tracking options
        self.use_vertical_tracking = True
        self.use_rotation_tracking = True
        self.use_horizontal_tracking = True
        self.use_distance_tracking = True

        # distance between object and drone
        self.dist_setpoint = 200
        self.area_setpoint = 20

        self.cx = 0
        self.cy = 0

        # processing frequency (to spare CPU time)
        self.cycle_counter = 1
        self.cycle_activation = 5

        # Kalman estimator scale factors
        self.kvscale = 6
        self.khscale = 4
        self.distscale = 3

        # Is the user changing target point?
        self.change_tp = False
        # (track_id) ask user whether to follow the available target 
        self.question_id = 1

        self.wt = safethread.SafeThread(target=self.__worker)
        self.wt.start()
        self.at = safethread.SafeThread(target=self.choose_id_initiate)


    def set_default_distance(self,DISTANCE=100):
        self.dist_setpoint = DISTANCE


    def set_default_area(self,AREA_RATIO=13):
        self.area_setpoint = AREA_RATIO


    def set_tracking(self, HORIZONTAL=False, VERTICAL=True, DISTANCE=True, ROTATION=True):
        self.use_vertical_tracking = VERTICAL
        self.use_horizontal_tracking = HORIZONTAL
        self.use_distance_tracking = DISTANCE
        self.use_rotation_tracking = ROTATION


    def set_image_to_process(self, img):
        """Image to process

        Args:
            img (nxmx3): RGB image
        """
        self.img = img


    def set_detection_periodicity(self,PERIOD=10):
        """
        Sets detection periodicity.
        Args:
            PERIOD (int, optional): detection timebase ~5ms*PERIOD. Defaults to 10.
        """
        self.cycle_activation = PERIOD


    def safety_limiter(self,leftright,fwdbackw,updown,yaw, SAFETYLIMIT=30):
        """
        Implement a safety limiter if values exceed defined threshold

        Args:
            leftright ([type]): control value for left right
            fwdbackw ([type]): control value for forward backward
            updown ([type]): control value up down
            yaw ([type]): control value rotation
        """
        val = np.array([leftright,fwdbackw,updown,yaw])

        val[val>=SAFETYLIMIT] = (SAFETYLIMIT)
        val[val<=-SAFETYLIMIT] = -(SAFETYLIMIT)

        return val[0],val[1],val[2],val[3]


    def __worker(self):
        """Worker thread to process command / detections
        """
        # time base
        self.ticker.wait(0.005)

        # process image, command tello
        if self.img is not None and self.cycle_counter % self.cycle_activation == 0:

            dist = 0
            vy = 0
            vx,rx = 0,0
            # work on a local copy
            img = self.img.copy()

            # detect face
            tp,det,ID = self.model.detect_track(img)
            self.det = det
            self.tp = tp
            self.ID = ID

            if self.change_tp and self.ID == []:
                self.at.start()
        
            if  len(det) > 0:
                
                # init estimators
                if self.track == False:
                    h,w = img.shape[:2]
                    self.cx = w//2
                    self.cy = h//2
                    self.kf.init(self.cx,self.cy)

                    # compute init 'area', ignore x dimension
                    self.kfarea.init(1,tp[1])
                    self.track = True

                if self.follow:
                    # process corrections, compute delta between two objects
                    _,cp = self.kf.predictAndUpdate(self.cx,self.cy,True)

                    # calculate delta over 2 axis
                    mvx = -int((cp[0]-tp[0])//self.kvscale)
                    mvy = int((cp[1]-tp[1])//self.khscale)

                    if self.use_distance_tracking:
                        # use detection y value to estimate object distance
                        obj_y = tp[2]

                        _, ocp = self.kfarea.predictAndUpdate(1, obj_y, True)

                        dist = int((ocp[1]-self.dist_setpoint)//self.distscale)

                    # Fill out variables to be sent in the tello command
                    # don't combine horizontal and rotation
                    if self.use_horizontal_tracking:
                        rx = 0
                        vx = mvx
                    if self.use_rotation_tracking:
                        vx = 0
                        rx = mvx

                    if self.use_vertical_tracking:
                        vy = mvy

                    # limit signals if is the case, could save your tello
                    vx,dist,vy,rx = self.safety_limiter(vx,dist,vy,rx,SAFETYLIMIT=40)

                    if self.flying:
                        self.tello.send_rc_control(int(vx), -int(dist), int(vy), int(rx))

            else:
                self.det = None
                if self.follow and self.flying:
                    # no detection, keep position
                    self.tello.send_rc_control(0, 0, 0, 0)
                
        self.cycle_counter +=1

                 
    def draw_detections(self,img, HUD=True):
        """Draw detections on an image

        Args:
            img (nxmx3): RGB image array
            HUD (boolean): overlay tello information
        """
        sizef = 1
        typef = cv2.FONT_HERSHEY_SIMPLEX
        color = [255,0,0]
        sizeb = 2

        if img is not None:
            
            h,w = img.shape[:2]

            if HUD:
                battery = self.tello.get_battery()
                height = self.tello.get_height()
                tof = self.tello.get_distance_tof()
                cv2.putText(img,'Battery: '+str(battery),(10, h-10),typef,sizef,color,sizeb)
                
                cv2.putText(img,'Height: '+str(height),(10, h-40),typef,sizef,color,sizeb)

                cv2.putText(img,'Tof: '+str(tof),(10, h-70),typef,sizef,color,sizeb)

                if self.follow:
                    cv2.putText(img,'Mode: 1',(w-310, 40),typef,sizef,color,sizeb)
                else:
                    cv2.putText(img,'Mode: 0',(w-310, 40),typef,sizef,color,sizeb)
                cv2.putText(img,'Mode 1 - Drone Auto Following',(w-310, 85),typef,0.6,color,sizeb)
                cv2.putText(img,'Mode 0 - Manual Control',(w-310, 65),typef,0.6,color,sizeb)

            # handle detection visualization
            if self.det is not None:            
                for val, track_ID in zip(self.det, self.ID):
                    # draw blue bbox to ask user whether follow the detection, other detections draw green bbox
                    if self.change_tp and track_ID == self.question_id:
                        cv2.rectangle(img,(val[0],val[1]),(val[0]+val[2],val[1]+val[3]),[0,0,255],2)
                        cv2.putText(img,'ID: ' + str(track_ID),(val[0],val[1]-10),typef,0.5,[0,0,255],2)
                        cv2.putText(img,'TRACK?',(val[0],val[1]-30),typef,0.7,[0,0,255],2)
                    else:
                        cv2.rectangle(img,(val[0],val[1]),(val[0]+val[2],val[1]+val[3]),[0,255,0],2)
                        cv2.putText(img,'ID: ' + str(track_ID),(val[0],val[1]-10),typef,0.5,[0,255,0],2)
                        
                if self.follow:
                    cv2.circle(img,(self.tp[0],self.tp[1]),3,[0,0,255],-1)
                    cv2.circle(img,(int(w/2),int(h/2)),4,[0,255,0],1)
                    cv2.line(img,(int(w/2),int(h/2)),(self.tp[0],self.tp[1]),[0,255,0],2)

    def stop_thread(self):
        self.wt.stop()
        self.wt.join()

    # Switch available target to ask user whether to follow
    def choose_id(self, initiate=False, lr=False, confirm=False):

        # Call when user start choosing potential target, ask first available target
        if initiate:
            if len(self.ID) > 0:
                self.question_id = self.ID[0]

        # Change ID via ascending or descending    
        if lr == 'RIGHT':
            for index, detected_ID in enumerate(self.ID):
                if detected_ID == self.question_id and index != len(self.ID) - 1:
                    self.question_id = self.ID[index + 1]
                    break
                self.question_id = self.ID[0]
        elif lr == 'LEFT':
            for index, detected_ID in enumerate(self.ID):
                if detected_ID == self.question_id and index != 0:
                    self.question_id = self.ID[index - 1]
                    break
                elif detected_ID == self.question_id and index == 0:
                    self.question_id = self.ID[-1]
                    break
                self.question_id = self.ID[0]
        
        # call when user confirm target to be followed
        if confirm:
            self.model.chosen_id = self.question_id

    def choose_id_initiate(self):
        if len(self.ID) > 0:
            self.question_id = self.ID[0]
            self.at.stop()
            
