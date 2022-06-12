import cv2
import numpy as np
import matplotlib.pyplot as plt

camera_matrix = np.array([[502.0652345727587, 0, 950.9190392476233],
                          [0, 502.1233514734862, 558.721201377333],
                          [0, 0, 1]])
distort = np.array(
    [1.546085974289742, 0.4834651376442347, -8.693842508843272e-05, 4.996418181759835e-05, 0.01160422887216484,
     1.891680709296835, 0.952186428460311, 0.08281205878591356, 0, 0, 0, 0, 0, 0])


class cameras_model():
    def __init__(self, camera_num):
        self.camera_ID = []
        self.camera_data = {}
        self.cameras = []
        self.map_probability = {}
        self.map1, self.map2 = cv2.initUndistortRectifyMap(camera_matrix, distort, np.eye(3), camera_matrix,
                                                           (1920, 1080),
                                                           cv2.CV_16SC2)
        for i in range(camera_num):
            self.camera_ID.append(2+i*2)
            cap = cv2.VideoCapture(self.camera_ID[i])
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
            cap.set(cv2.CAP_PROP_FPS, 30)
            key = 0
            ret=False
            while not (key == 13):
                ret,frame=cap.read()
                if ret:
                    cv2.imshow(str(i), frame)
                    key = cv2.waitKey(30)
                else:
                    print("open camera %s failed" % self.camera_ID[i])
                    break
            cv2.destroyWindow(str(i))
            name = input("input the name of this camera")
            self.camera_data[name] = i
            self.cameras.append(cap)

    def show_camera(self, camera_name):
        if camera_name in self.camera_data:
            i = self.camera_data[camera_name]
            ret, frame = self.cameras[i].read()
            frame = cv2.remap(frame, self.map1, self.map2, cv2.INTER_LINEAR)

            cv2.imshow(camera_name, frame)
        else:
            print("no camera called " + camera_name)
        return frame

    def generate_probability_map(self, camera_name, frame):
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        map_probability = np.zeros_like(frame_gray)
        if camera_name == "front":
            now_point = 0
            start_point = now_point
            while True:
                pix_vec = frame_gray[now_point, :]
                if np.any(pix_vec):
                    now_point = now_point + 1
                else:
                    break
                if now_point == frame.shape[0]:
                    break
            short_vec = np.linspace(0.5, 1, np.abs(now_point - start_point))
            if frame.shape[1] % 2 == 0:
                long_vec = np.linspace(0.5, 1, int(frame.shape[1] / 2))
                long_vec_flip = np.linspace(1, 0.5, int(frame.shape[1] / 2))
                long_vec = np.append(long_vec, long_vec_flip)
            else:
                long_vec = np.linspace(0.5, 1, int((frame.shape[1] - 1) / 2))
                long_vec_flip = np.linspace(1, 0.5, (frame.shape[1] - 1) / 2)
                long_vec = np.append(long_vec, 1)
                long_vec = np.append(long_vec, long_vec_flip)
            matrix_probability = np.dot(short_vec.reshape(-1, 1), long_vec.reshape(1, -1))
            map_probability = map_probability[0:frame.shape[0] - now_point, :]
            matrix_probability = np.row_stack((matrix_probability, map_probability))
            map_probability = matrix_probability
            map_probability = map_probability * frame_gray
            map_probability = np.nan_to_num(map_probability / frame_gray)
            self.map_probability["front"] = map_probability
            return matrix_probability
        elif camera_name == "left":
            now_point = 0
            start_point = now_point
            while True:
                pix_vec = frame_gray[:, now_point]
                if np.any(pix_vec):
                    now_point = now_point + 1
                else:
                    break
            short_vec = np.linspace(0.5, 1, np.abs(now_point - start_point))
            if frame.shape[0] % 2 == 0:
                long_vec = np.linspace(0.5, 1,int(frame.shape[0] / 2))
                long_vec_flip = np.linspace(1, 0.5, int(frame.shape[0] / 2))
                long_vec = np.append(long_vec, long_vec_flip)
            else:
                long_vec = np.linspace(0.5, 1, int((frame.shape[0] - 1) / 2))
                long_vec_flip = np.linspace(1, 0.5, (frame.shape[0] - 1) / 2)
                long_vec = np.append(long_vec, 1)
                long_vec = np.append(long_vec, long_vec_flip)
            matrix_probability = np.dot(long_vec.reshape(-1, 1), short_vec.reshape(1, -1))
            map_probability=map_probability[:,0:frame.shape[1]-now_point]
            map_probability=np.column_stack((matrix_probability,map_probability))
            map_probability = map_probability * frame_gray
            map_probability = np.nan_to_num(map_probability / frame_gray)
            self.map_probability["left"] = map_probability
            return matrix_probability
        elif camera_name == "right":
            now_point = frame.shape[1] - 1
            start_point = now_point
            while True:
                pix_vec = frame_gray[:, now_point]
                if np.any(pix_vec):
                    now_point = now_point - 1
                else:
                    break
            short_vec = np.linspace(1, 0.5, np.abs(now_point - start_point))
            if frame.shape[0] % 2 == 0:
                num_rol = int(frame.shape[0] / 2)
                long_vec = np.linspace(0.5, 1, num_rol)
                long_flip = np.linspace(1, 0.5, num_rol)
                long_vec = np.append(long_vec, long_flip)
            else:
                long_vec = np.linspace(1, 0.5, int((frame.shape[0] - 1) / 2))
                long_vec_flip = np.flip(long_vec)
                long_vec = np.append(long_vec, 1)
                long_vec = np.append(long_vec, long_vec_flip)
            matrix_probability = np.dot(long_vec.reshape(-1, 1), short_vec.reshape(1, -1))
            map_probability = map_probability[:, 0:now_point + 1]
            map_probability = np.column_stack((map_probability, matrix_probability))
            # map_probability[:, now_point:start_point] = matrix_probability

            map_probability = map_probability * frame_gray
            map_probability = np.nan_to_num(map_probability / frame_gray)
            self.map_probability["right"] = map_probability
            return matrix_probability
        else:
            now_point = frame.shape[0] - 1
            start_point = now_point
            while True:
                pix_vec = frame_gray[now_point, :]
                if np.any(pix_vec):
                    now_point = now_point - 1
                else:
                    break
            short_vec = np.linspace(1, 0.5, np.abs(now_point - start_point))
            if frame.shape[1] % 2 == 0:
                long_vec = np.linspace(0.5, 1, int(frame.shape[1] / 2))
                long_vec_flip=np.linspace(1,0.5,int(frame.shape[1] / 2))
                long_vec = np.append(long_vec, long_vec_flip)
            else:
                long_vec = np.linspace(0.5, 1, int((frame.shape[1] - 1) / 2))
                long_vec_flip = np.flip(long_vec)
                long_vec = np.append(long_vec, 1)
                long_vec = np.append(long_vec, long_vec_flip)
            matrix_probability = np.dot(short_vec.reshape(-1, 1), long_vec.reshape(1, -1))
            map_probability=map_probability[0:now_point+1,:]
            map_probability=np.row_stack((map_probability,matrix_probability))
            map_probability = map_probability * frame_gray
            map_probability = np.nan_to_num(map_probability / frame_gray)
            self.map_probability["back"] = map_probability
            return matrix_probability

    def generate_sum_map_probability(self):
        sum_map_probability = np.zeros_like(self.map_probability["front"])
        for key in self.map_probability:
            sum_map_probability = sum_map_probability + self.map_probability[key]
        for key in self.map_probability:
            self.map_probability[key] = np.nan_to_num(self.map_probability[key] / sum_map_probability)

    def show_stich(self, frames):
        frame = np.zeros_like(frames["front"])
        for i in frames:
            frame[:, :, 0] = frame[:, :, 0] + frames[i][:, :, 0] * self.map_probability[i]
            frame[:, :, 1] = frame[:, :, 1] + frames[i][:, :, 1] * self.map_probability[i]
            frame[:, :, 2] = frame[:, :, 2] + frames[i][:, :, 2] * self.map_probability[i]
        cv2.imshow("stich", frame)

    def show_pro_frame(self, frame, name):
        map_pro = self.map_probability[name]
        frame[:, :, 0] = frame[:, :, 0] * map_pro
        frame[:, :, 1] = frame[:, :, 1] * map_pro
        frame[:, :, 2] = frame[:, :, 2] * map_pro
        cv2.imwrite(name + ".jpg", frame)
        cv2.imshow(name, frame)
