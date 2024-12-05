import cv2
import mediapipe as mp
import numpy as np
import pyaudio
import threading
import tkinter as tk
from tkinter import ttk


def closest_element(arr, target):
    min_diff = float('inf')
    closest = None
    for num in arr:
        diff = abs(num - target)
        if diff < min_diff:
            min_diff = diff
            closest = num
    return closest


class TrumpAccordion:
    notes = [
        "A3", "A#3", "B3", "C4", "C#4", "D4", "D#4", "E4", "F4", "F#4", "G4", "G#4", "A4", "A#4", "B4", "C5", "C#5", "D5", "D#5", "E5", "F5", "F#5", "G5", "G#5", "A5", "A#5", "B5", "C6"
    ]
    note_freqs = [
        220, 233, 247, 262, 277, 294, 311, 330, 349, 370, 392, 415, 440, 466, 494, 523, 554, 587, 622, 659, 698, 740, 784, 831, 880, 932, 988, 1047
    ]

    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=2, min_detection_confidence=0.7)

        self.p = pyaudio.PyAudio()
        self.sample_rate = 44100
        self.volume = 0.0
        self.current_frequency = 440
        self.target_frequency = 440
        self.frequency_smoothing_rate = 0.1

        self.stream = self.p.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=self.sample_rate,
            output=True,
            frames_per_buffer=1024
        )

        self.phase = 0
        self.hands_detected = False
        self.current_note = "A4"

        self.audio_thread = threading.Thread(
            target=self.generate_audio, daemon=True)
        self.audio_thread.start()

    def get_note_from_frequency(self, freq):
        index = (np.abs(np.array(self.note_freqs) - freq)).argmin()
        return self.notes[index]

    def calculate_finger_distance(self, hand1, hand2):
        thumb1_tip = hand1.landmark[4]
        index1_tip = hand1.landmark[8]
        thumb2_tip = hand2.landmark[4]
        index2_tip = hand2.landmark[8]

        distance = np.sqrt(
            (thumb1_tip.x - thumb2_tip.x)**2 +
            (thumb1_tip.y - thumb2_tip.y)**2 +
            (index1_tip.x - index2_tip.x)**2 +
            (index1_tip.y - index2_tip.y)**2
        )
        return distance

    def calculate_average_y_position(self, hand1, hand2):
        index1_tip = hand1.landmark[8]
        index2_tip = hand2.landmark[8]
        return (index1_tip.y + index2_tip.y) / 2

    def process_frame(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)

        h, w, _ = frame.shape

        # 修复手部检测逻辑
        if results.multi_hand_landmarks and len(results.multi_hand_landmarks) == 2:
            self.hands_detected = True
            hand1, hand2 = results.multi_hand_landmarks

            # 双手食指之间的距离决定音量
            finger_distance = self.calculate_finger_distance(hand1, hand2)
            # 手指越张开，音量越大
            self.volume = max(0, min(1, finger_distance / 0.5))

            # 双手食指y轴平均位置决定音高
            avg_y_position = self.calculate_average_y_position(hand1, hand2)
            self.target_frequency = closest_element(
                self.note_freqs, self.note_freqs[0] + (
                    1 - avg_y_position) * (self.note_freqs[-1] - self.note_freqs[0]))

            self.current_note = self.get_note_from_frequency(
                self.target_frequency)

            # fmt: off
            cv2.putText(
              frame,
              f'{self.current_note} {self.target_frequency:.0f}Hz Vol:{self.volume:.2f}',
              (int(w/2) - 150, int(h/2)),
              cv2.FONT_HERSHEY_SIMPLEX,
              1.5,
              (0, 255, 0),
              4
           )
            # fmt: on
        else:
            self.hands_detected = False

        self.draw_scale(frame)

        return frame

    def draw_scale(self, frame):
        for i in range(len(self.note_freqs)):
            freq = self.note_freqs[i]
            note = self.notes[i]
            y = int(frame.shape[0] * (1 - (freq - self.note_freqs[0]
                                           ) / (self.note_freqs[-1] - self.note_freqs[0])))
            cv2.line(frame, (0, y), (frame.shape[1], y), (0, 0, 255), 1)
            cv2.putText(frame, note, (5, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    def generate_audio(self):
        while True:
            if not self.hands_detected:
                self.volume = 0
                self.phase = 0
                self.current_frequency = 440
                self.target_frequency = 440

            # Smooth frequency transition
            self.current_frequency += (self.target_frequency -
                                       self.current_frequency) * self.frequency_smoothing_rate

            t_array = np.linspace(0, 1024/self.sample_rate, 1024, False)
            samples = np.sin(
                2 * np.pi * self.current_frequency * t_array + self.phase)
            samples *= self.volume * 0.5

            audio_data = samples.astype(np.float32)
            self.stream.write(audio_data.tobytes())

            self.phase += 2 * np.pi * self.current_frequency * 1024 / self.sample_rate
            self.phase %= 2 * np.pi

    def run(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            processed_frame = self.process_frame(frame)

            cv2.imshow('川普手风琴', processed_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.on_closing()

    def on_closing(self):
        self.root.destroy()
        self.cap.release()
        cv2.destroyAllWindows()
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()
        exit()


def main():
    accordion = TrumpAccordion()
    accordion.run()


if __name__ == "__main__":
    main()
