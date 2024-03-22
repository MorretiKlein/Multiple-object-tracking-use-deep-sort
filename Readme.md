# Multiple object tracking 

## Define folder

* `test_tepm`: .
    * `data`(small dataset) and `human dataset`(large dataset): install from https://www.kaggle.com/datasets/klein2111/data-human-2 and run preprocess data
    I only train with small dataset and it's nt really good. Depending on the problem, I think it's a good idea to get actual data
* `run`: save best.pt. 
## description 
Object detection: detect people , need `git clone https://github.com/ultralytics/ultralytics.git` 
    using YOLO v8

* Multiple object tracking:  track each person throughout a single video --> tracklets
    is crucial to build an accurate gallery of unique identities.
    Object detection only is applied separately on each frame and we create tracker to encode this temporal(tạm thời)
    information by linking detections of each object throughout the whole video.
    we can using tracker: Deep Sort. (combination of 2 techniques: a Kalman filter and a visual appearance encoder)
* Link tracklets of the same person
    re-identification is a much more challenging task due to severe appearance changes across different camera viewpoints.
    Differences in camera position, color balance, resolution, body pose… make the same identities look very different.

* Kalman filter predicts the position of each pedestrian in the next frame based on the detected bounding boxes in the previous frames.
    However, irregular paths or occlusion (tắc nghẽn) cause techniques purely can fail Kalman.
    If can control or solving irregular oath by camera, we can onlu use Kalman
* Visual appearance encoder: is convolution neural network that extracts a feature vector from an image.
    The here image is bounding box image, In the latent space imposed by these feature vectors, two extracted feature corresponding to the same identity are likely to be closer than different identities. (using distance metric - cosine similiraties distance)

* ` I follow the reposities https://github.com/ZQPei/deep_sort_pytorch.git , also use pytorch.But I shortened certain functions to better conform to the conventions of my project, which utilize the xyxy format.`
`I've worked on this repository to deepen my understanding of Deep Sort in preparation for Multiple Target Multiple Camera Tracking (MTMCT). I plan to revisit it and make further modifications to enhance accuracy.`
## to run and check repository
you have to environment have torch with cuda, sklearn, cv2, scipy

## Result:
<img src="output_video.gif" />
