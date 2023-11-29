import cv2
import time
import sys
import math
import mediapipe as mp
import numpy as np

from sortedcontainers import SortedList
from collections import deque
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.python.solutions.pose import PoseLandmark
from mediapipe.framework.formats import landmark_pb2

def draw_landmarks_on_image(rgb_image, detection_result):
    pose_landmarks_list = detection_result.pose_landmarks
    annotated_image = np.copy(rgb_image)

    nose = pose_landmarks_list[0][PoseLandmark.NOSE]
    l_ear = pose_landmarks_list[0][PoseLandmark.LEFT_EAR]
    r_ear = pose_landmarks_list[0][PoseLandmark.RIGHT_EAR]

    norm_nose = landmark_pb2.NormalizedLandmark(x=nose.x, y=nose.y, z=nose.z)
    norm_l_ear = landmark_pb2.NormalizedLandmark(x=l_ear.x, y=l_ear.x, z=l_ear.x)
    norm_r_ear = landmark_pb2.NormalizedLandmark(x=r_ear.x, y=r_ear.x, z=r_ear.x)

    image_rows, image_cols, _ = annotated_image.shape
    nose_px = solutions.drawing_utils._normalized_to_pixel_coordinates(nose.x, nose.y, image_cols, image_rows)
    l_ear_px = solutions.drawing_utils._normalized_to_pixel_coordinates(l_ear.x, l_ear.y, image_cols, image_rows)

    annotated_image = cv2.ellipse(annotated_image, nose_px, (100,100), 30, 0, 360, (0, 0, 0), -1)

    # Loop through the detected poses to visualize.
    for idx in range(len(pose_landmarks_list)):
        pose_landmarks = pose_landmarks_list[idx]

        # Draw the pose landmarks.
        pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        pose_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
        ])
        solutions.drawing_utils.draw_landmarks(
            annotated_image,
            pose_landmarks_proto,
            solutions.pose.POSE_CONNECTIONS,
            solutions.drawing_styles.get_default_pose_landmarks_style())
    return annotated_image

def avg(list):
    return sum(list) / len(list)

def append_tuple(tuple, x, y):
    tuple[0].append(x)
    tuple[1].append(y)

def print_point_queue_info(point_queue, name):
    print(name)
    q_x = point_queue[0]
    q_y = point_queue[1]
    print('minimum = ({}, {})'.format(min(q_x), min(q_y)))
    print('maximum = ({}, {})'.format(max(q_x), max(q_y)))
    print('average = ({}, {})\n'.format(avg(q_x), avg(q_y)))

model_path = './pose_landmarker_full.task'

def main():
    detection_result_list = []
    image_result_list = []
    delay_queue = deque(maxlen=300)

    shoulder_tuple = (deque(maxlen=300), deque(maxlen=300))
    elbow_tuple = (deque(maxlen=300), deque(maxlen=300))
    wrist_tuple = (deque(maxlen=300), deque(maxlen=300))

    angle_list = SortedList()

    counter, fps = 0, 0

    shoulder_rom = True

    start_time = time.time()

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Visualization parameters
    row_size = 20  # pixels
    left_margin = 24  # pixels
    text_color = (255, 255, 0)  # red
    font_size = 1
    font_thickness = 1
    fps_avg_frame_count = 10

    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    PoseLandmarkerResult = mp.tasks.vision.PoseLandmarkerResult
    VisionRunningMode = mp.tasks.vision.RunningMode

    def result_callback(result: PoseLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
        result.timestamp_ms = timestamp_ms

        right_shoulder = result.pose_landmarks[0][PoseLandmark.RIGHT_SHOULDER]
        right_elbow = result.pose_landmarks[0][PoseLandmark.RIGHT_ELBOW]
        joint = result.pose_landmarks[0][PoseLandmark.RIGHT_SHOULDER]
        right_wrist = result.pose_landmarks[0][PoseLandmark.RIGHT_WRIST]
        nose = result.pose_landmarks[0][PoseLandmark.NOSE]

        if shoulder_rom == False:
            joint = result.pose_landmarks[0][PoseLandmark.RIGHT_ELBOW]

        right_shoulder_xy = [right_shoulder.x, right_shoulder.y]
        joint_xy = [joint.x, joint.y]
        down_xy = [joint.x, 1]
        right_elbow_xy = [right_elbow.x, right_elbow.y]
        right_wrist_xy = [right_wrist.x, right_wrist.y]

        append_tuple(shoulder_tuple, right_shoulder.x, right_shoulder.y)
        append_tuple(elbow_tuple, right_elbow.x, right_elbow.y)
        append_tuple(wrist_tuple, right_wrist.x, right_wrist.y)

        print_point_queue_info(shoulder_tuple, 'Shoulder:')
        print_point_queue_info(elbow_tuple, 'Elbow:')
        print_point_queue_info(wrist_tuple, 'Wrist:')

        jt_rwdist = math.dist(joint_xy, right_wrist_xy)
        jt_dwdist = math.dist(joint_xy, down_xy)
        rw_dwdist = math.dist(right_wrist_xy, down_xy)

        angle = math.degrees(math.acos((jt_rwdist**2 + jt_dwdist**2 - rw_dwdist**2) / (2 * jt_rwdist * jt_dwdist)))

        rs_n_x_dist = nose.x - right_shoulder.x
        rs_rw_x_dist = right_wrist.x - right_shoulder.x
        if (np.sign(rs_n_x_dist) != np.sign(rs_rw_x_dist)):
            angle = 360. - angle

        print('arm angle: {}\n'.format(angle))
        angle_list.add(angle)
        result.angle = angle

        print('Right Shoulder: [ {} , {} , {} ]'.format(right_shoulder.x, right_shoulder.y, right_shoulder.z))
        print('Right Elbow: [ {} , {} , {} ]'.format(right_elbow.x, right_elbow.y, right_elbow.z))
        print('Right Wrist: [ {} , {} , {} ]\n'.format(right_wrist.x, right_wrist.y, right_wrist.z))

        delay_queue.append(int(time.time() * 1000) - timestamp_ms)
        detection_result_list.clear()
        detection_result_list.append(result)
        image_result_list.clear()
        image_result_list.append(output_image)

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.LIVE_STREAM,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        result_callback=result_callback)
    
    landmarker = vision.PoseLandmarker.create_from_options(options)

    while cap.isOpened():
        ret, img = cap.read()

        if not ret:
           sys.exit('ERROR: Unable to read from webcam. Please verify your webcam settings.')

        counter += 1

        # convert default bgr image capture to rgb
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_img)

        landmarker.detect_async(mp_img, int(time.time() * 1000))

        current_frame = mp_img.numpy_view()
        current_frame = cv2.cvtColor(current_frame, cv2.COLOR_RGB2BGR)

        # Calculate the FPS
        if counter % fps_avg_frame_count == 0:
            end_time = time.time()
            fps = fps_avg_frame_count / (end_time - start_time)
            start_time = time.time()

        if detection_result_list and image_result_list:
            current_frame = image_result_list[0].numpy_view()
            current_frame = cv2.cvtColor(current_frame, cv2.COLOR_RGB2BGR)
            # Show the FPS
            fps_text = 'FPS = {:.1f}'.format(fps)
            text_location = (left_margin, row_size)
            cv2.putText(current_frame, fps_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                        font_size, text_color, font_thickness)
            vis_img = draw_landmarks_on_image(current_frame, detection_result_list[0])

            # Show the delay
            delay = 0
            if len(delay_queue) > 0:
                delay = sum(delay_queue) / len(delay_queue)
            delay_text = 'Delay = {:1f} ms'.format(delay)
            text_location = (left_margin, 2 * row_size)
            cv2.putText(vis_img, delay_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                        font_size, text_color, font_thickness)

            # Show the angle
            angle_text = 'Shoulder Angle = {:3f}'.format(detection_result_list[0].angle)
            text_location = (left_margin, 3 * row_size)
            cv2.putText(vis_img, angle_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                        font_size, text_color, font_thickness)

            # Show the max angle
            angle_text = 'Average Max Angle = {:3f}'.format(avg(angle_list[-100:]))
            text_location = (left_margin, 4 * row_size)
            cv2.putText(vis_img, angle_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                        font_size, text_color, font_thickness)

            # Show the current joint
            angle_text = 'Shoulder = {:3f}'.format(shoulder_rom)
            text_location = (left_margin, 5 * row_size)
            cv2.putText(vis_img, angle_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                        font_size, text_color, font_thickness)
            cv2.imshow('pose detection', vis_img)

        else:
            cv2.imshow('pose detection', current_frame)

        keypress = cv2.waitKey(1)

        if keypress == 114:
            angle_list.clear()

        # Switch which joint to measure ROM for when space is pressed
        if keypress == 32:
            shoulder_rom = not shoulder_rom

        # End when esc is pressed
        if keypress == 27:
            break

    landmarker.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(e)