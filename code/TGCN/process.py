import json
import os
import time
import torch

def compute_difference(x):
    diff = []
    for i, xx in enumerate(x):
        temp = []
        for j, xxx in enumerate(x):
            if i != j:
                temp.append(xx - xxx)
        diff.append(temp)
    return diff

body_pose_exclude = {9, 10, 11, 22, 23, 24, 12, 13, 14, 19, 20, 21}

# Specify the video capture details
video_id = '01'
frame_start = 1
frame_end = 100

save_to = os.path.join('c:/Users/Pc/Desktop/SilentTalk/code/TGCN/new_feature/', video_id)
if not os.path.exists(save_to):
    os.mkdir(save_to)

for frame_id in range(frame_start, frame_end + 1):
    frame_id = 'image_{}'.format(str(frame_id).zfill(5))
    ft_path = os.path.join(save_to, frame_id + '_ft.pt')
    if not os.path.exists(ft_path):
        try:
            pose_content = json.load(open(os.path.join('c:/Users/Pc/Desktop/SilentTalk/data/pose_per_individual_videos',
                                                       video_id, frame_id + '_keypoints.json')))["people"][0]
        except IndexError:
            continue

        body_pose = pose_content["pose_keypoints_2d"]
        left_hand_pose = pose_content["hand_left_keypoints_2d"]
        right_hand_pose = pose_content["hand_right_keypoints_2d"]

        body_pose.extend(left_hand_pose)
        body_pose.extend(right_hand_pose)

        x = [v for i, v in enumerate(body_pose) if i % 3 == 0 and i // 3 not in body_pose_exclude]
        y = [v for i, v in enumerate(body_pose) if i % 3 == 1 and i // 3 not in body_pose_exclude]

        x = 2 * ((torch.FloatTensor(x) / 256.0) - 0.5)
        y = 2 * ((torch.FloatTensor(y) / 256.0) - 0.5)

        x_diff = torch.FloatTensor(compute_difference(x)) / 2
        y_diff = torch.FloatTensor(compute_difference(y)) / 2

        zero_indices = (x_diff == 0).nonzero()
        orient = y_diff / x_diff
        orient[zero_indices] = 0

        xy = torch.stack([x, y]).transpose_(0, 1)
        ft = torch.cat([xy, x_diff, y_diff, orient], dim=1)

        torch.save(ft, ft_path)

print('Processing complete for the video capture.')