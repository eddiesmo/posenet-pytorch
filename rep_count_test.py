import pickle

import cv2
import numpy as np

import posenet

coord_hist_pickle_file = 'keypoint_coord_hist.p'


def draw_pose(kp_coord_hist):
    height = 720
    width = 1280
    blank_image = np.zeros((height, width, 3), np.uint8)

    display_image = posenet.draw_skel_and_kp(
        blank_image, pose_scores, keypoint_scores, keypoint_coords,
        min_pose_score=0.25, min_part_score=0.25)

    cv2.imshow('posenet', display_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


def main():
    kp_coord_hist = pickle.load(open(coord_hist_pickle_file), 'rb')


if __name__ == '__main__':
    main()
