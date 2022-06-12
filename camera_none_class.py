import numpy as np
import cv2


def generate_sum_map_probability(map_probability):
    sum_map_probability = np.zeros_like(map_probability["front"])
    for key in map_probability:
        sum_map_probability = sum_map_probability + map_probability[key]
    for key in map_probability:
        map_probability[key] = np.nan_to_num(map_probability[key] / sum_map_probability)
    return map_probability


def show_stich(frames, map_probability):
    frame = np.zeros_like(frames["front"])
    for i in frames:
        frame[:, :, 0] = frame[:, :, 0] + frames[i][:, :, 0] * map_probability[i]
        frame[:, :, 1] = frame[:, :, 1] + frames[i][:, :, 1] * map_probability[i]
        frame[:, :, 2] = frame[:, :, 2] + frames[i][:, :, 2] * map_probability[i]
    cv2.imshow("stich", frame)


def generate_probability_map(camera_name, frame):
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
        return map_probability
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
            long_vec = np.linspace(0.5, 1, int(frame.shape[0] / 2))
            long_vec_flip = np.linspace(1, 0.5, int(frame.shape[0] / 2))
            long_vec = np.append(long_vec, long_vec_flip)
        else:
            long_vec = np.linspace(0.5, 1, int((frame.shape[0] - 1) / 2))
            long_vec_flip = np.linspace(1, 0.5, (frame.shape[0] - 1) / 2)
            long_vec = np.append(long_vec, 1)
            long_vec = np.append(long_vec, long_vec_flip)
        matrix_probability = np.dot(long_vec.reshape(-1, 1), short_vec.reshape(1, -1))
        map_probability = map_probability[:, 0:frame.shape[1] - now_point]
        map_probability = np.column_stack((matrix_probability, map_probability))
        map_probability = map_probability * frame_gray
        map_probability = np.nan_to_num(map_probability / frame_gray)
        return map_probability
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
        return map_probability
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
            long_vec_flip = np.linspace(1, 0.5, int(frame.shape[1] / 2))
            long_vec = np.append(long_vec, long_vec_flip)
        else:
            long_vec = np.linspace(0.5, 1, int((frame.shape[1] - 1) / 2))
            long_vec_flip = np.flip(long_vec)
            long_vec = np.append(long_vec, 1)
            long_vec = np.append(long_vec, long_vec_flip)
        matrix_probability = np.dot(short_vec.reshape(-1, 1), long_vec.reshape(1, -1))
        map_probability = map_probability[0:now_point + 1, :]
        map_probability = np.row_stack((map_probability, matrix_probability))
        map_probability = map_probability * frame_gray
        map_probability = np.nan_to_num(map_probability / frame_gray)
        return map_probability
