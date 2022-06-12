import cv2
import numpy as np

# 设置寻找亚像素角点的参数，采用的停止准则是最大循环次数30和最大误差容限0.001
criteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 30, 0.001)
objp = np.zeros((8 * 11, 3), np.float32)
objp[:, :2] = np.mgrid[0:8, 0:11].T.reshape(-1, 2)
obj_points = []  # 存储3D点
img_points = []  # 存储2D点

camera_matrix = np.array([[502.0652345727587, 0, 950.9190392476233],
                          [0, 502.1233514734862, 558.721201377333],
                          [0, 0, 1]])
distort = np.array(
    [1.546085974289742, 0.4834651376442347, -8.693842508843272e-05, 4.996418181759835e-05, 0.01160422887216484,
     1.891680709296835, 0.952186428460311, 0.08281205878591356, 0, 0, 0, 0, 0, 0])

if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    cap.set(cv2.CAP_PROP_FPS, 30)
    cv2.VideoWriter("part_slot.mp4", -1 , 30, (1920,1080), False)
    key_str = '0'
    while not (key_str == 10):
        ret, frame = cap.read()
        cv2.imshow("f", frame)
        key_str = cv2.waitKey(30)
    cv2.waitKey()
    cv2.destroyWindow("f")
    map1, map2 = cv2.initUndistortRectifyMap(camera_matrix, distort, np.eye(3), camera_matrix, (1920, 1080),
                                             cv2.CV_16SC2)
    frame = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)
    cv2.imshow("undistort",frame)
    cv2.waitKey()
    cv2.destroyWindow("undistort")
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    size = gray.shape[::-1]
    ret, corners = cv2.findChessboardCorners(gray, (8, 11), None)
    print(ret)
    if ret:
        obj_points.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)  # 在原角点的基础上寻找亚像素角点
        # print(corners2)
        if [corners2]:
            img_points.append(corners2)
        else:
            img_points.append(corners)
        point_seq = [0, 7, 80, 87]
        src_point = []
        dst_point = []
        for i in range(0, 4):
            point = img_points[0][point_seq[i]]
            src_point.append(point)
            point = obj_points[0][point_seq[i]].reshape(1, -1)
            point = point[:, 0:2]
            point = point * 10
            dst_point.append(point)
        src_point = np.array(src_point)
        dst_point = np.array(dst_point)
        homo_matrix = cv2.findHomography(src_point, dst_point, cv2.RANSAC)
        frame1 = cv2.warpPerspective(frame, homo_matrix[0], (1920, 1080))
        cv2.imshow("f", frame1)
        cv2.waitKey()
