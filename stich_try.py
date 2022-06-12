import cv2
import numpy as np
import camera_none_class
import chess

camera_matrix = np.array([[502.0652345727587, 0, 950.9190392476233],
                          [0, 502.1233514734862, 558.721201377333],
                          [0, 0, 1]])
distort = np.array(
    [1.546085974289742, 0.4834651376442347, -8.693842508843272e-05, 4.996418181759835e-05, 0.01160422887216484,
     1.891680709296835, 0.952186428460311, 0.08281205878591356, 0, 0, 0, 0, 0, 0])

map1, map2 = cv2.initUndistortRectifyMap(camera_matrix, distort, np.eye(3), camera_matrix,
                                         (1920, 1080),
                                         cv2.CV_16SC2)

homo_matrix = {}
direction = ["front", "back", "left", "right"]
all_points = chess.generate_obj_points(3.675, 4.625 - 3.675)
direction_points = {}
points_arry = {}
pic_points = {}
frame_undistort = {}
frame_ver = {}
frame_ori = {}
map_pro = {}
for i in direction:
    point_vec = []
    direction_points[i] = all_points[i]
    point_vec.append(direction_points[i]['point_left_up'])
    point_vec.append(direction_points[i]['point_right_up'])
    point_vec.append(direction_points[i]['point_left_down'])
    point_vec.append(direction_points[i]['point_right_down'])
    points_arry[i] = np.array(point_vec)
    pic_points[i] = chess.obj_to_pic(points_arry[i])
frame_ori["front"] = cv2.imread("f/0.jpg")
frame_ori["left"] = cv2.imread("l/0.jpg")
frame_ori["right"] = cv2.imread("r/0.jpg")
frame_ori["back"] = cv2.imread("b/0.jpg")
for i in direction:
    frame_undistort[i] = cv2.remap(frame_ori[i], map1, map2, cv2.INTER_LINEAR)
    cv2.imshow(i,frame_undistort[i])
    cv2.waitKey()
    ret, homo_matrix[i] = chess.get_trans_matrix(frame_undistort[i], (6, 4), pic_points[i])
for i in direction:
    frame_ver[i] = cv2.warpPerspective(frame_undistort[i], homo_matrix[i], (750, 1000))
    cv2.imshow(i,frame_ver[i])
    cv2.waitKey()
    map_pro[i] = camera_none_class.generate_probability_map(i, frame_ver[i])

map_pro = camera_none_class.generate_sum_map_probability(map_pro)


camera_none_class.show_stich(frame_ver,map_pro)
cv2.waitKey()
