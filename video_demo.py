import argparse
import time

import cv2
import torch
import pickle
import posenet

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=int, default=101)
parser.add_argument('--cam_id', type=int, default=0)
parser.add_argument('--cam_width', type=int, default=1280)
parser.add_argument('--cam_height', type=int, default=720)
parser.add_argument('--scale_factor', type=float, default=0.7125)
args = parser.parse_args()


def get_videowriter(out_name='output.avi', fps=20, out_size=(1280, 720)):
    fourcc_code = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(out_name, fourcc_code, fps, out_size)
    return out


class PoseHistory:
    def __init__(self):
        class History:
            pose_scores = []
            keypoint_scores = []
            keypoint_coords = []

        self.history = History()

    def update_history(self, pose_scores, keypoint_scores, keypoint_coords):
        self.history.pose_scores.append(pose_scores)
        self.history.keypoint_scores.append(keypoint_scores)
        self.history.keypoint_coords.append(keypoint_coords)

    @staticmethod
    def save_history_to_file(self, file_name: str = 'pose_history.p'):
        pickle.dump(file_name, 'wb')


def main():
    model = posenet.load_model(args.model)
    model = model.cuda()
    output_stride = model.output_stride

    start = time.time()
    input_vid_name = '2020-11-01-174051.webm'
    cap = cv2.VideoCapture(input_vid_name)
    out = get_videowriter()

    frame_count = 0
    keypoint_coords_hist = []
    while cap.isOpened():
        print(f'frame {frame_count}')
        retval, frame = cap.read()
        if not retval:
            break

        input_image, display_image, output_scale = posenet.process_input(frame, args.scale_factor)

        # res, input_image, display_image, output_scale = posenet.read_video_cap(
        #     cap, scale_factor=args.scale_factor, output_stride=output_stride)

        with torch.no_grad():
            input_image = torch.Tensor(input_image).cuda()

            heatmaps_result, offsets_result, displacement_fwd_result, displacement_bwd_result = model(input_image)

            pose_scores, keypoint_scores, keypoint_coords = posenet.decode_multiple_poses(
                heatmaps_result.squeeze(0),
                offsets_result.squeeze(0),
                displacement_fwd_result.squeeze(0),
                displacement_bwd_result.squeeze(0),
                output_stride=output_stride,
                max_pose_detections=10,
                min_pose_score=0.25)

        keypoint_coords *= output_scale

        # update_pose_history(pose_scores, keypoint_scores, keypoint_coords)
        # chosen_instance = np.argmax(pose_scores)
        # keypoint_coords_hist.append(keypoint_coords[0, :])

        display_image = posenet.draw_skel_and_kp(
            display_image, pose_scores, keypoint_scores, keypoint_coords,
            min_pose_score=0.25, min_part_score=0.25)

        out.write(display_image)
        cv2.imshow('posenet', display_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        frame_count += 1

    pickle.dump()
    cap.release()
    cv2.destroyAllWindows()
    out.release()


if __name__ == "__main__":
    main()
