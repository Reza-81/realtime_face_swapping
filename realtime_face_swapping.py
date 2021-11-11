import cv2
import numpy as np
import time
import mediapipe as mp

# creat object to detect face landmarks
mp_draw = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1)

# index of best landmarks
list_inex_of_landmarks_around_face = {0, 1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19, 20, 21, 31
                                      , 33, 37, 38, 39, 40, 41, 42, 46, 52, 53, 54, 55, 58, 61, 63, 64, 65
                                      , 66, 67, 70, 72, 73, 74, 76, 77, 79, 80, 81, 82, 84, 85, 86, 87, 88
                                      , 89, 90, 91, 93, 94, 95, 96, 97, 102, 103, 105, 107, 109, 113, 115
                                      , 127, 131, 132, 133, 134, 136, 141, 144, 145, 146, 148, 149, 150, 151
                                      , 152, 153, 154, 155, 157, 158, 159, 160, 161, 162, 163, 164, 168, 172
                                      , 173, 174, 176, 178, 179, 180, 181, 183, 184, 185, 191, 195, 197, 198
                                      , 218, 221, 222, 223, 224, 225, 226, 228, 229, 230, 231, 232, 233, 234
                                      , 236, 240, 242, 244, 246, 249, 250, 251, 261, 263, 267, 268, 269, 270
                                      , 271, 272, 276, 282, 283, 284, 285, 288, 291, 293, 294, 295, 296, 297
                                      , 300, 302, 303, 304, 307, 309, 310, 312, 314, 315, 316, 317, 318, 319
                                      , 320, 321, 323, 324, 325, 328, 331, 332, 334, 336, 338, 342, 344, 356
                                      , 360, 361, 362, 363, 365, 370, 373, 374, 375, 377, 378, 379, 380, 381
                                      , 382, 384, 385, 386, 387, 388, 389, 390, 397, 398, 399, 400, 402, 403
                                      , 404, 405, 407, 408, 409, 413, 415, 420, 438, 441, 442, 443, 444, 445
                                      , 446, 448, 449, 450, 451, 452, 453, 454, 456, 460, 462, 463, 464}


def find_coordinates_of_face_landmarks(image):
    ih, iw, ic = image.shape
    coordinates_of_face_landmarks = []
    for face_lms in face_mesh.process(image).multi_face_landmarks:
        for index in list_inex_of_landmarks_around_face:
            x, y = int(face_lms.landmark[index].x * iw), int(face_lms.landmark[index].y * ih)
            coordinates_of_face_landmarks.append([x, y])
    return coordinates_of_face_landmarks

# fps
p_time, c_time = 0, 0

# read image
image_path = input("enter your image path: ")
image_1 = cv2.imread(image_path)

#read video from camera
cap = cv2.VideoCapture(0)

# Face 1 landmarks
coordinates_of_face_1_landmarks = find_coordinates_of_face_landmarks(image_1)
coordinates_of_face_1_landmarks_array = np.array(coordinates_of_face_1_landmarks, np.int32)

# creat mask_face_1 from face 1 and bitwise_and
image_1_gray = cv2.cvtColor(image_1, cv2.COLOR_BGR2GRAY)
mask_face_1 = np.zeros_like(image_1_gray)
convexhull_1 = cv2.convexHull(coordinates_of_face_1_landmarks_array)
cv2.fillConvexPoly(mask_face_1, convexhull_1, 255)
face_image_1 = cv2.bitwise_and(image_1, image_1, mask=mask_face_1)

# Delaunay triangulation for face 1
rect = cv2.boundingRect(convexhull_1)
subdiv = cv2.Subdiv2D(rect)
subdiv.insert(coordinates_of_face_1_landmarks)
triangles_face_1 = subdiv.getTriangleList()
triangles_face_1 = np.array(triangles_face_1, dtype=np.int32)

# get index of triangles_face_1
triangles_indexes = []
for t in triangles_face_1:
    pt1 = (t[0], t[1])
    pt2 = (t[2], t[3])
    pt3 = (t[4], t[5])
    index_pt1 = np.where((coordinates_of_face_1_landmarks_array == pt1).all(axis=1))[0][0]
    index_pt2 = np.where((coordinates_of_face_1_landmarks_array == pt2).all(axis=1))[0][0]
    index_pt3 = np.where((coordinates_of_face_1_landmarks_array == pt3).all(axis=1))[0][0]
    triangles_indexes.append([index_pt1, index_pt2, index_pt3])

# read video from camera and put face 1 on it
while True:
    try:
        _, image_2 = cap.read()
        image_2 = cv2.flip(image_2, 1)
        img_2_gray = cv2.cvtColor(image_2, cv2.COLOR_BGR2GRAY)
        img_2_new_face = np.zeros_like(image_2)

        # Face 2
        coordinates_of_face_2_landmarks = find_coordinates_of_face_landmarks(image_2)
        coordinates_of_face_2_landmarks_array = np.array(coordinates_of_face_2_landmarks, np.int32)

        # Triangulation of both faces
        for triangle in triangles_indexes:
            # Triangulation of the first face
            tr1_pt1 = coordinates_of_face_1_landmarks[triangle[0]]
            tr1_pt2 = coordinates_of_face_1_landmarks[triangle[1]]
            tr1_pt3 = coordinates_of_face_1_landmarks[triangle[2]]
            triangle_from_face_1 = np.array([tr1_pt1, tr1_pt2, tr1_pt3], np.int32)
            rect_1 = cv2.boundingRect(triangle_from_face_1)
            (x, y, w, h) = rect_1
            cropped_triangle = image_1[y: y + h, x: x + w]
            cropped_triangle_1_mask = np.zeros((h, w), np.uint8)
            triangle_from_face_1_cropped = np.array([[tr1_pt1[0] - x, tr1_pt1[1] - y],
                                                     [tr1_pt2[0] - x, tr1_pt2[1] - y],
                                                     [tr1_pt3[0] - x, tr1_pt3[1] - y]], np.int32)
            cv2.fillConvexPoly(cropped_triangle_1_mask, triangle_from_face_1_cropped, 255)

            # Triangulation of second face
            tr2_pt1 = coordinates_of_face_2_landmarks[triangle[0]]
            tr2_pt2 = coordinates_of_face_2_landmarks[triangle[1]]
            tr2_pt3 = coordinates_of_face_2_landmarks[triangle[2]]
            triangle2 = np.array([tr2_pt1, tr2_pt2, tr2_pt3], np.int32)
            rect_2 = cv2.boundingRect(triangle2)
            (x, y, w, h) = rect_2
            cropped_triangle2_mask = np.zeros((h, w), np.uint8)
            triangle_from_face_2_cropped = np.array([[tr2_pt1[0] - x, tr2_pt1[1] - y],
                                                     [tr2_pt2[0] - x, tr2_pt2[1] - y],
                                                     [tr2_pt3[0] - x, tr2_pt3[1] - y]], np.int32)
            cv2.fillConvexPoly(cropped_triangle2_mask, triangle_from_face_2_cropped, 255)

            # Warp triangles_face_1
            M = cv2.getAffineTransform(np.float32(triangle_from_face_1_cropped), np.float32(triangle_from_face_2_cropped))
            warped_triangle = cv2.warpAffine(cropped_triangle, M, (w, h))
            warped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle, mask=cropped_triangle2_mask)

            # Reconstructing destination face
            image_2_new_face_rect_area = img_2_new_face[y: y + h, x: x + w]
            image_2_new_face_rect_area_gray = cv2.cvtColor(image_2_new_face_rect_area, cv2.COLOR_BGR2GRAY)
            _, mask_triangles_designed = cv2.threshold(image_2_new_face_rect_area_gray, 1, 255, cv2.THRESH_BINARY_INV)
            warped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle, mask=mask_triangles_designed)
            image_2_new_face_rect_area = cv2.add(image_2_new_face_rect_area, warped_triangle)
            img_2_new_face[y: y + h, x: x + w] = image_2_new_face_rect_area

        # Face swapped (putting 1st face into 2nd face)
        image_2_face_mask = np.zeros_like(img_2_gray)
        convexhull2 = cv2.convexHull(coordinates_of_face_2_landmarks_array)
        image_2_head_mask = cv2.fillConvexPoly(image_2_face_mask, convexhull2, 255)
        image_2_face_mask = cv2.bitwise_not(image_2_head_mask)
        image_2_head_noface = cv2.bitwise_and(image_2, image_2, mask=image_2_face_mask)
        result = cv2.add(image_2_head_noface, img_2_new_face)

        # find fps
        c_time = time.time()
        fps = 1 / (c_time - p_time)
        p_time = c_time

        # showing final image
        cv2.putText(result, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 0), 3)
        cv2.imshow("result", result)
        key = cv2.waitKey(1)

    except Exception as e:
        print(e)

cap.release()
cv2.destroyAllWindows()