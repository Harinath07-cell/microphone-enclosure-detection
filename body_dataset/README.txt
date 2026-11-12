BODY DATASET (YOLO FORMAT)

Source video: BODY.mp4
Resolution: 832x464
FPS: 59.94
Duration: 47.45 seconds
Saved images: 444
Split: train=313, val=86, test=45

Class:
  0 body

Structure:
 images/train, images/val, images/test : extracted frames
 labels/train, labels/val, labels/test : YOLO annotation files (currently empty)
 data.yaml : YOLO dataset config

ANNOTATION STATUS:
Images are extracted and organized. Bounding boxes still need to be annotated.
For each image, draw a box around the visible enclosure body and save a matching .txt file
in the corresponding labels folder. YOLO line format:
0 x_center y_center width height
(all coordinates normalized to 0..1).

This is a body-only video, so use class 0 (body) for visible body objects.
