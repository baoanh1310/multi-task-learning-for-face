# multi-task-learning-for-face
Smile - Emotion - Gender - Age Multi-task learning using BKNet / ResNet

## Installation
```console
  pip install -r requirements.txt
```

## How to use it
```console
  git clone https://github.com/baoanh1310/multi-task-learning-for-face.git
  cd multi-task-learning-for-face
  python video_stream_demo.py --prototxt ./face_detector/deploy.prototxt.txt --model ./face_detector/res10_300x300_ssd_iter_140000.caffemodel
```

## Optional

You can try ResNet model by following below steps:

```console
  cd multi_resnet
  python video_stream_demo.py --prototxt ../face_detector/deploy.prototxt.txt --model ../face_detector res10_300x300_ssd_iter_140000.caffemodel
```
