import cv2
import time
import sys
import mediapipe as mp
import numpy as np

from collections import deque
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.python.solutions.pose import PoseLandmark
from mediapipe.framework.formats import landmark_pb2

def draw_landmarks_on_image(rgb_image, detection_result):
    pose_landmarks_list = detection_result.pose_landmarks
    annotated_image = np.copy(rgb_image)

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

def append_triple(triple, x, y, z):
    triple[0].append(x)
    triple[1].append(y)
    triple[2].append(z)

def print_point_queue_info(point_queue, name):
    print(name)
    q_x = point_queue[0]
    q_y = point_queue[1]
    q_z = point_queue[2]
    print('minimum = ({}, {}, {})'.format(min(q_x), min(q_y), min(q_z)))
    print('maximum = ({}, {}, {})'.format(max(q_x), max(q_y), max(q_z)))
    print('average = ({}, {}, {})\n'.format(avg(q_x), avg(q_y), avg(q_z)))

model_path = './pose_landmarker_full.task'

def main():
    detection_result_list = []
    image_result_list = []
    delay_queue = deque(maxlen=300)

    shoulder_triple = (deque(maxlen=300), deque(maxlen=300), deque(maxlen=300))
    elbow_triple = (deque(maxlen=300), deque(maxlen=300), deque(maxlen=300))
    wrist_triple = (deque(maxlen=300), deque(maxlen=300), deque(maxlen=300))

    counter, fps = 0, 0

    start_time = time.time()

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Visualization parameters
    row_size = 20  # pixels
    left_margin = 24  # pixels
    text_color = (0, 0, 255)  # red
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

        #result.pose_landmarks[0][PoseLandmark.RIGHT_SHOULDER].z = 1

        right_shoulder = result.pose_landmarks[0][PoseLandmark.RIGHT_SHOULDER]
        right_elbow = result.pose_landmarks[0][PoseLandmark.RIGHT_ELBOW]
        right_wrist = result.pose_landmarks[0][PoseLandmark.RIGHT_WRIST]

        # Calculate angle based on https://stackoverflow.com/a/35178910
        right_shoulder_xyz = np.array([right_shoulder.x, right_shoulder.y, right_shoulder.z])
        right_elbow_xyz = np.array([right_elbow.x, right_elbow.y, right_elbow.z])
        right_wrist_xyz = np.array([right_wrist.x, right_wrist.y, right_wrist.z])

        append_triple(shoulder_triple, right_shoulder.x, right_shoulder.y, right_shoulder.z)
        append_triple(elbow_triple, right_elbow.x, right_elbow.y, right_elbow.z)
        append_triple(wrist_triple, right_wrist.x, right_wrist.y, right_wrist.z)

        print_point_queue_info(shoulder_triple, 'Shoulder:')
        print_point_queue_info(elbow_triple, 'Elbow:')
        print_point_queue_info(wrist_triple, 'Wrist:')

        vec1 = right_wrist_xyz - right_elbow_xyz
        vec2 = right_shoulder_xyz - right_elbow_xyz
        cosine_angle = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        angle = np.arccos(cosine_angle)
        print('elbow angle: {}\n'.format(np.degrees(angle)))
        result.angle = np.degrees(angle)


        #rw.x = 0
        #rw.y = 0

        print('Right Shoulder: [ {} , {} , {} ]'.format(right_shoulder.x, right_shoulder.y, right_shoulder.z))
        print('Right Elbow: [ {} , {} , {} ]'.format(right_elbow.x, right_elbow.y, right_elbow.z))
        print('Right Wrist: [ {} , {} , {} ]\n'.format(right_wrist.x, right_wrist.y, right_wrist.z))

        #print('Right Shoulder: {}'.format(result.pose_landmarks[0][PoseLandmark.RIGHT_SHOULDER]))
        #print('Right Elbow: {}'.format(result.pose_landmarks[0][PoseLandmark.RIGHT_ELBOW]))
        #print('Right Wrist: {}'.format(result.pose_landmarks[0][PoseLandmark.RIGHT_WRIST]))

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

        #img = cv2.flip(img, 1)

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
            cv2.imshow('pose detection', vis_img)

            # Show the angle
            angle = 0
            angle_text = 'Elbow Angle = {:3f}'.format(detection_result_list[0].angle)
            text_location = (left_margin, 3 * row_size)
            cv2.putText(vis_img, angle_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                        font_size, text_color, font_thickness)
            cv2.imshow('pose detection', vis_img)
        else:
            cv2.imshow('pose detection', current_frame)

        if cv2.waitKey(1) == 27:
            break

    landmarker.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()