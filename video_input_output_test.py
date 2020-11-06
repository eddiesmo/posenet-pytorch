import argparse
import time

import cv2
import torch

import posenet


def get_videowriter(out_name='output.avi', fps=30.0, out_size=(1280, 720)):
    fourcc_code = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(out_name, fourcc_code, fps, out_size)
    return out


def main():
    cap = cv2.VideoCapture('jumping-jacks.mp4')
    out = get_videowriter()

    frame_count = 0
    while cap.isOpened():
        print(f'frame {frame_count}')
        ret, frame = cap.read()
        if not ret:
            break

        out.write(frame)
        cv2.imshow('posenet', frame)
        frame_count += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    out.release()


if __name__ == "__main__":
    main()
