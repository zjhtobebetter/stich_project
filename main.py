# This project is used for transforming photos from fisheye camera to ground view and stitch different cameras
import cv2
import numpy as np
import camera
import chess

if __name__ == '__main__':
#     cap=cv2.VideoCapture(8)
#     cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
#     cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
#     cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
#     cap.set(cv2.CAP_PROP_FPS, 30)
#     frame=cap.read()
#
#     while True:
#         ret,frame=cap.read()
#         cv2.imshow("8",frame)
#         cv2.waitKey(30)

    homo_matrix = {}
    direction = ["front", "back", "left", "right"]
    my_camera = camera.cameras_model(4)
    all_points = chess.generate_obj_points(3.675, 4.625 - 3.675)
    direction_points = {}
    points_arry = {}
    pic_points = {}
    frame_undistort = {}
    frame_ver = {}
    for i in direction:
        point_vec = []
        direction_points[i] = all_points[i]
        point_vec.append(direction_points[i]['point_left_up'])
        point_vec.append(direction_points[i]['point_right_up'])
        point_vec.append(direction_points[i]['point_left_down'])
        point_vec.append(direction_points[i]['point_right_down'])
        points_arry[i] = np.array(point_vec)
        pic_points[i] = chess.obj_to_pic(points_arry[i])

    for i in my_camera.camera_data:
        key = 0
        while not key == 13:
            frame_undistort[i] = my_camera.show_camera(i)
            key = cv2.waitKey(30)
        key = 0
        while not key == 13:
            frame_undistort[i] = my_camera.show_camera(i)
            key = cv2.waitKey(30)
            ret, homo_matrix[i] = chess.get_trans_matrix(frame_undistort[i], (6, 4), pic_points[i])
            if ret:
                break
        cv2.destroyWindow(i)

    print(homo_matrix)

    for i in my_camera.camera_data:
        key = 0
        while not (key == 13):
            frame_undistort[i] = my_camera.show_camera(i)
            frame_ver[i] = cv2.warpPerspective(frame_undistort[i], homo_matrix[i], (750, 1000))
            cv2.imshow(i + "_ver", frame_ver[i])
            key = cv2.waitKey(30)
        cv2.destroyWindow(i + "_ver")
        test_m = my_camera.generate_probability_map(i, frame_ver[i])

    my_camera.generate_sum_map_probability()
    key = 0
    while not key == 13:
        for i in my_camera.camera_data:
            frame_undistort[i] = my_camera.show_camera(i)
            frame_ver[i] = cv2.warpPerspective(frame_undistort[i], homo_matrix[i], (750, 1000))
        my_camera.show_stich(frame_ver)
        key = cv2.waitKey(30)

