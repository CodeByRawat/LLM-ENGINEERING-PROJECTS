
import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import random
import time

st.set_page_config(page_title="Cartoon Eyes Tracker", layout="centered")
st.title("👀 Cartoon Eyes That Track You")

class CartoonEyes(VideoTransformerBase):
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=False)
        self.blink_timer = time.time()
        self.is_blinking = False
        self.left_eye_pos = (220, 200)
        self.right_eye_pos = (420, 200)

    def maybe_blink(self):
        if time.time() - self.blink_timer > random.uniform(3, 6):
            self.is_blinking = True
            self.blink_timer = time.time()
        elif self.is_blinking and time.time() - self.blink_timer > 0.15:
            self.is_blinking = False

    def draw_stylized_eye(self, canvas, center, iris_offset=(0, 0), blink=False):
        eye_radius = 60
        pupil_radius = 20
        if blink:
            cv2.ellipse(canvas, center, (eye_radius, 10), 0, 0, 360, (180, 180, 180), -1)
            return
        cv2.circle(canvas, center, eye_radius, (220, 220, 220), -1)
        cv2.circle(canvas, center, eye_radius, (50, 50, 50), 2)
        pupil_center = (center[0] + iris_offset[0], center[1] + iris_offset[1])
        cv2.circle(canvas, pupil_center, pupil_radius, (0, 0, 0), -1)
        cv2.circle(canvas, (pupil_center[0] - 5, pupil_center[1] - 5), 5, (200, 200, 200), -1)
        cv2.circle(canvas, pupil_center, pupil_radius + 5, (90, 90, 180), 2)

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        h, w = img.shape[:2]
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        canvas = np.ones((400, 640, 3), dtype=np.uint8) * 255
        self.maybe_blink()

        iris_offset_left = (0, 0)
        iris_offset_right = (0, 0)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark

            def get_px(idx):
                lm = landmarks[idx]
                return int(lm.x * w), int(lm.y * h)

            face_center = get_px(1)

            def calc_offset(eye_pos):
                dx = face_center[0] - w // 2
                dy = face_center[1] - h // 2
                dx = int(np.clip(dx * 0.20, -25, 25))
                dy = int(np.clip(dy * 0.20, -25, 25))
                return dx, dy

            iris_offset_left = calc_offset(self.left_eye_pos)
            iris_offset_right = calc_offset(self.right_eye_pos)

        self.draw_stylized_eye(canvas, self.left_eye_pos, iris_offset_left, self.is_blinking)
        self.draw_stylized_eye(canvas, self.right_eye_pos, iris_offset_right, self.is_blinking)

        return canvas

# Launch the Streamlit app
webrtc_streamer(key="eye-tracker", video_transformer_factory=CartoonEyes)
