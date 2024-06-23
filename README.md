# tello_drone_human_tracking
Detect and track human using DJI Tello drone.

### Features

- Mode 0 for manual drone control using keyboard
- Mode 1 for auto drone human tracking
- Switching target to follow when multiple human is detected in mode 1
- Keyboard input:
  - Common mode: t/l/v/m - take off/land/toggle video recording/toggle mode
  - Mode 0: w/s/a/d/left/right/up/down - up/down/ccw rotation/cw rotation/left/right/forward/backward 
  - Mode 1: c/left/right/enter - toggle target switching panel/switch target id descendingly/switch target id ascendingly/confirm target
- Video recorded by drone will be saved as video.avi

<p align="center"> 
    <img src="./info/demo.gif" alt="400" width="400">
</p>

### Run Program

```
usage: main.py [-h] [-vsize VSIZE] [-th TH] [-tv TV] [-td TD] [-tr TR]

options:
  -h, --help    show this help message and exit
  -vsize VSIZE  Video size received from tello as "(width, height)"
  -th TH        Horizontal tracking
  -tv TV        Vertical tracking
  -td TD        Distance tracking
  -tr TR        Rotation tracking

e.g. python main.py -vsize "(600,400)" -th True -tr False
```

### Requirements

- Download Anaconda and create an environment using command line list in requirement.txt or create similar environment using other ways
- DJI Tello drone

### Reference

1. https://github.com/fvilmos/tello_object_tracking
2. https://github.com/damiafuentes/DJITelloPy
3. https://github.com/ultralytics/ultralytics
4. https://github.com/nwojke/deep_sort
5. https://github.com/computervisioneng/object-tracking-yolov8-deep-sort
6. ChatGPT
 
