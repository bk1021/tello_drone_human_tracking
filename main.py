import signal
import cv2
import argparse
import pygame
import numpy as np
import time
from djitellopy import Tello
from utils.followobject import FollowObject

# Speed of the drone in manual control
S = 40
# Frames per second of the pygame event handling
FPS = 120

class FrontEnd():
    def __init__(self, args):
        # Init pygame for keyboard event handling
        pygame.init()
        pygame.display.set_caption("Tello Video Stream")
        self.screen = pygame.display.set_mode([960, 720])

        pygame.time.set_timer(pygame.USEREVENT + 1, 1000 // FPS)

        self.tello = Tello()
        self.fobj = FollowObject(self.tello)
        self.fobj.set_tracking(HORIZONTAL=args.th, VERTICAL=args.tv, DISTANCE=args.td, ROTATION=args.tr)

        # init drone
        self.tello.connect()
        self.tello.set_speed(10)
        self.tello.streamoff()
        self.tello.streamon()

        self.frame_read = self.tello.get_frame_read()
        self.h, self.w, _ = self.frame_read.frame.shape
        if args.vsize is not None:
            args.vsize = tuple(map(int, args.vsize.strip('()').split(',')))
            self.vsize = args.vsize
            self.video = cv2.VideoWriter('video.avi', cv2.VideoWriter_fourcc(*'XVID'), 30, self.vsize)
        else:
            self.vsize = (self.w, self.h)
            self.video = cv2.VideoWriter('video.avi', cv2.VideoWriter_fourcc(*'XVID'), 30, self.vsize)

        # Drone velocities between -100~100
        self.for_back_velocity = 0
        self.left_right_velocity = 0
        self.up_down_velocity = 0
        self.yaw_velocity = 0

        # init conditions 
        self.send_rc_control = False
        self.writevideo = False
        # drone following
        self.follow = False
        # Is the user changing target point?
        self.change_tp = False

    def run(self):       
        should_run = True
        while should_run:
            frame = self.frame_read.frame

            for event in pygame.event.get():
                if event.type == pygame.USEREVENT + 1:
                    if not self.follow: self.update()
                elif event.type == pygame.QUIT:
                    should_run = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        should_run = False
                    else:
                        self.keydown(event.key)
                elif event.type == pygame.KEYUP:
                    self.keyup(event.key)
                    
            if self.frame_read.stopped:
                break
            
            self.image_processing(frame)
            pygame.display.update()

            time.sleep(1 / FPS)

        self.end()

    def keydown(self, key):
        if key == pygame.K_UP:
            self.for_back_velocity = S
        elif key == pygame.K_DOWN:
            self.for_back_velocity = -S
        elif key == pygame.K_LEFT:
            self.left_right_velocity = -S
        elif key == pygame.K_RIGHT:
            self.left_right_velocity = S
        elif key == pygame.K_w:
            self.up_down_velocity = S
        elif key == pygame.K_s:
            self.up_down_velocity = -S
        elif key == pygame.K_a:
            self.yaw_velocity = -S
        elif key == pygame.K_d:
            self.yaw_velocity = S


    def keyup(self, key):
        if key in [pygame.K_UP, pygame.K_DOWN]:
            self.for_back_velocity = 0
        elif key in [pygame.K_LEFT, pygame.K_RIGHT]:
            self.left_right_velocity = 0
            if self.change_tp and key == pygame.K_LEFT: 
                self.fobj.choose_id(lr='LEFT')
            elif self.change_tp: 
                self.fobj.choose_id(lr='RIGHT')
        elif key in [pygame.K_w, pygame.K_s]:
            self.up_down_velocity = 0
        elif key in [pygame.K_a, pygame.K_d]:
            self.yaw_velocity = 0
        elif key == pygame.K_t:
            self.tello.takeoff()
            self.send_rc_control = True
            self.fobj.flying = True
        elif key == pygame.K_l:
            self.tello.land()
            self.send_rc_control = False
            self.fobj.flying = False
        elif key == pygame.K_v:
            self.writevideo = not self.writevideo
        elif key == pygame.K_m:
            self.follow = not self.follow
            self.fobj.follow = not self.fobj.follow
            if not self.follow:
                self.change_tp = False
                self.fobj.change_tp = False
        elif key == pygame.K_c:
            if self.follow:
                self.change_tp = not self.change_tp
                self.fobj.change_tp = not self.fobj.change_tp
                if not self.change_tp:
                    self.fobj.choose_id(initiate=True)
        elif key == pygame.K_RETURN:
            if self.change_tp:
                self.fobj.choose_id(confirm=True)
                self.change_tp = False
                self.fobj.change_tp = False


    def update(self):
        if self.send_rc_control:
            self.tello.send_rc_control(self.left_right_velocity, self.for_back_velocity, self.up_down_velocity, self.yaw_velocity)


    def image_processing(self, frame):
        if frame is not None:
            self.fobj.set_image_to_process(frame)

            # work on a local copy
            imghud = frame.copy()
            self.fobj.draw_detections(imghud)
            video_frame = cv2.cvtColor(imghud, cv2.COLOR_BGR2RGB)
            video_frame = cv2.resize(video_frame, self.vsize)

            if self.writevideo:
                self.video.write(video_frame)

            # Add recording icon on display screen
            if self.writevideo:
                cv2.putText(imghud,'Recording',(10, 40),cv2.FONT_HERSHEY_COMPLEX,1,[255,0,0],2)

            imghud = np.rot90(imghud)
            imghud = np.flipud(imghud)

            pygame_imghud = pygame.surfarray.make_surface(imghud)
            self.screen.blit(pygame_imghud, (0, 0))


    def end(self):
        self.fobj.stop_thread()
        self.tello.streamoff()
        self.tello.end()
        pygame.quit()
        self.video.release()


def signal_handler(sig, frame):
    raise Exception("Signal received, stopping...")


def main():
    parser = argparse.ArgumentParser(description='Tello Object tracker. keys: t-takeoff, l-land, v-video, w-up, s-down, a-ccw rotate, d-cw rotate\n')
    parser.add_argument('-vsize', type=str, help='Video size received from tello as "(width, height)"', default=None)
    parser.add_argument('-th', type=bool, help='Horizontal tracking', default=False)
    parser.add_argument('-tv', type=bool, help='Vertical tracking', default=True)
    parser.add_argument('-td', type=bool, help='Distance tracking', default=True)
    parser.add_argument('-tr', type=bool, help='Rotation tracking', default=True)

    args = parser.parse_args()

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    frontend = FrontEnd(args)

    frontend.run()
    print('PROGRAM END.')


if __name__ == '__main__':
    main()