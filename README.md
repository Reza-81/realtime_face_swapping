# Real Time Face swapper

## A face swapper using **opencv** and **mediapipe** packages written in python

This project is based on this YouTube [video](https://youtu.be/dK-KxuPi768), but instead of using the **dlib** package, I used **mediapipe** for the face detection process.

## How to run the program

1. first you need to install **opencv** and **mediapipe**

   `pip install opencv-python`

   `pip install mediapipe`

2. then run **realtime_face_swapping.py**

   `python realtime_face_swapping.py`

## This is how the program works

1. first with **opencv** we read the first face image.
2. then by using **mediapipe** we can find the landmarks on the first face image.
3. using **opencv** and the specified landmarks, we can create some triangles on the first face image.
4. we do the same to the second image that we read from camera by using **opencv**.
5. after that we map the corresponding triangles from first face image to the second face image.
