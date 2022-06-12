import cv2
import numpy as np

# used to get transform matrix to vertical view

# 设置寻找亚像素角点的参数，采用的停止准则是最大循环次数30和最大误差容限0.001
criteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 30, 0.001)


def get_trans_matrix(frame, chess_size, obj_points):
    point_seq = [0, chess_size[0] - 1, chess_size[0] * (chess_size[1] - 1), chess_size[0] * chess_size[1] - 1]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img_points = []
    ret, corners = cv2.findChessboardCorners(gray, chess_size, None)
    if ret:
        corners2 = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
        if [corners2]:
            for i in range(4):
                img_points.append(corners2[point_seq[i]])
        else:
            for i in range(4):
                img_points.append(corners[point_seq[i]])
        img_points = np.array(img_points).reshape(4, 2)
        obj_points = np.array(obj_points)
        print("img")
        print(img_points)
        print("obj")
        obj_points = obj_points
        obj_points[:, 1] = obj_points[:, 1]
        print(obj_points)
        homo_matrix = cv2.findHomography(img_points, obj_points, cv2.RANSAC)
        return True, homo_matrix[0]
    else:
        return False, np.eye(3)


def rot_point(rad, point_zero, point_old):
    x0 = point_zero[0]
    y0 = point_zero[1]
    x = point_old[0]
    y = point_old[1]
    x_new = (x - x0) * np.cos(rad) - (y - y0) * np.sin(rad) + x0
    y_new = (x - x0) * np.sin(rad) + (y - y0) * np.cos(rad) + y0
    point_new = np.array([x_new, y_new])
    return point_new


def generate_obj_points(l1, l2, scale=50):
    # l1,l2 e the length from the back axis to the head and the chewei
    l1 = l1 * scale
    l2 = l2 * scale
    # define points of a chess
    chess_points = dict()
    chess_points["point_left_up"] = np.array([-0.5 * scale, 0.8 * scale])
    chess_points["point_right_up"] = np.array([0.5 * scale, 0.8 * scale])
    chess_points["point_left_down"] = np.array([-0.5 * scale, 0.2 * scale])
    chess_points["point_right_down"] = np.array([0.5 * scale, 0.2 * scale])
    # define zero point for each chess
    zero_points = dict()
    zero_points["zero_front"] = np.array([0, l1, 0])
    zero_points["zero_right"] = np.array([1.5 * scale, l1 - 2.3 * scale, -np.pi / 2])
    zero_points["zero_left"] = np.array([-1.1 * scale, l1 - 2.1 * scale, np.pi / 2])
    zero_points["zero_back"] = np.array([0.15*scale, -l2-1.1*scale, np.pi])
    all_ground_points = dict()
    for zero_name in zero_points:
        zero_point = zero_points[zero_name][0:2]
        rot_data = zero_points[zero_name][2]
        tem_points = dict()
        for chess_point_data in chess_points:
            chess_point = chess_points[chess_point_data]
            rot_chess_point = rot_point(rot_data, np.array([0, 0]), chess_point)
            rot_chess_point = rot_chess_point + zero_point
            tem_points[chess_point_data] = rot_chess_point
        all_ground_points[zero_name.replace("zero_", '')] = tem_points
    print(all_ground_points)
    return all_ground_points


def obj_to_pic(obj_points, scale=100, pic_shape=(1000, 750), backshaft=750):
    backshaft_location = np.array([pic_shape[1] / 2, backshaft])
    for i in range(4):
        obj_points[i, 0] = obj_points[i, 0] + backshaft_location[0]
        obj_points[i, 1] = backshaft_location[1] - obj_points[i, 1]
    return obj_points
