import cv2
import numpy as np
import mediapipe as mp
import time


class GesturePaint:
    def __init__(self, camera_index=0, fps=30, canvas_scale=1.5):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.75,
            min_tracking_confidence=0.75
        )
        self.mp_draw = mp.solutions.drawing_utils

        self.cap = cv2.VideoCapture(camera_index)
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Cannot open webcam")
        self.h, self.w = frame.shape[:2]

        self.canvas_h = int(self.h * canvas_scale)
        self.canvas_w = int(self.w * canvas_scale)

        self.canvas = np.zeros((self.canvas_h, self.canvas_w, 3), dtype=np.uint8)

        sw, m = 120, 15
        cols = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 255, 255)]
        self.palette = []
        x = m
        for c in cols:
            self.palette.append((x, self.canvas_h - m - sw, x + sw, self.canvas_h - m, c))
            x += sw + m

        self.brush_color = (0, 0, 255)
        self.brush_size = 8
        self.eraser_size = 30
        self.prev_pt = None
        self.prev_mode = None

        self.pinch_thresh = 50

        self.target_fps = fps
        self.prev_time = time.time()

        self.mode = 'draw'  # 'draw', 'erase', 'select'

    @staticmethod
    def _finger_up(lm, tip, pip):
        return lm[tip].y < lm[pip].y

    def _is_pinch(self, lm):
        thumb_tip = lm[4]
        index_tip = lm[8]
        distance = np.sqrt((thumb_tip.x - index_tip.x) ** 2 + (thumb_tip.y - index_tip.y) ** 2)
        return distance < self.pinch_thresh

    def run(self):
        while True:
            now = time.time()
            if now - self.prev_time < 1 / self.target_fps:
                continue
            self.prev_time = now

            ret, frame = self.cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)

            frame_resized = cv2.resize(frame, (self.canvas_w, self.canvas_h))
            rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            res = self.hands.process(rgb)

            ix = iy = None
            pinch = False
            hand_list = getattr(res, 'multi_hand_landmarks', None)

            if hand_list:
                lm = hand_list[0].landmark

                ix, iy = int(lm[8].x * self.canvas_w), int(lm[8].y * self.canvas_h)
                pinch = self._is_pinch(lm)
                cv2.circle(frame_resized, (ix, iy), 5, (0, 255, 0), -1)

                idx_up = self._finger_up(lm, 8, 6)
                mid_up = self._finger_up(lm, 12, 10)

                if self.mode != self.prev_mode:
                    self.prev_pt = None
                self.prev_mode = self.mode

                if self.mode == 'select' and pinch:
                    for x1, y1, x2, y2, c in self.palette:
                        if x1 < ix < x2 and y1 < iy < y2:
                            self.brush_color = c
                elif self.mode == 'erase' and idx_up and mid_up:
                    if self.prev_pt is None:
                        self.prev_pt = (ix, iy)
                    cv2.line(self.canvas, self.prev_pt, (ix, iy), (0, 0, 0), self.eraser_size)
                    self.prev_pt = (ix, iy)
                elif self.mode == 'draw' and idx_up and not mid_up:
                    if self.prev_pt is None:
                        self.prev_pt = (ix, iy)
                    cv2.line(self.canvas, self.prev_pt, (ix, iy), self.brush_color, self.brush_size)
                    self.prev_pt = (ix, iy)
                else:
                    self.prev_pt = None

                self.mp_draw.draw_landmarks(
                    frame_resized,
                    hand_list[0],
                    self.mp_hands.HAND_CONNECTIONS
                )

            else:
                self.prev_pt = None  # No hand → reset

            for x1, y1, x2, y2, c in self.palette:
                cv2.rectangle(frame_resized, (x1, y1), (x2, y2), c, -1)
                if self.mode == 'select' and pinch and ix and iy and x1 < ix < x2 and y1 < iy < y2:
                    cv2.rectangle(frame_resized, (x1, y1), (x2, y2), (255, 255, 255), 3)

            cv2.circle(frame_resized, (self.canvas_w - 60, 40), 20, self.brush_color, -1)
            cv2.putText(frame_resized, f"Mode: {self.mode.upper()}", (10, self.canvas_h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame_resized, f"Size: {self.eraser_size}", (self.canvas_w - 140, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            out = cv2.addWeighted(frame_resized, 0.5, self.canvas, 0.5, 0)
            cv2.imshow("Gesture Paint", out)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('c'):
                self.canvas[:] = 0
            elif key in [ord('+'), ord('=')]:
                self.brush_size = min(50, self.brush_size + 2)
            elif key in [ord('-'), ord('_')]:
                self.brush_size = max(2, self.brush_size - 2)
            elif key == ord('s'):
                self.mode = 'select'
            elif key == ord('d'):
                self.mode = 'draw'
            elif key == ord('e'):
                self.mode = 'erase'
            elif key == 27:  # ESC
                break

        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    GesturePaint().run()