import os
import torch
from torch.utils.data import DataLoader

from configs import Config
from sign_dataset import Sign_Dataset
from tgcn_model import GCN_muti_att


def test(model, data_loader):
    # Set model to evaluation mode
    model.eval()

    all_outputs = []

    with torch.no_grad():
        for batch_idx, data in enumerate(data_loader):
            # Move data to the device
            X = data['X'].cuda()

            # Perform forward pass
            output = model(X)

            # Store model's output
            all_outputs.append(output)

    # Concatenate outputs from all batches
    all_outputs = torch.cat(all_outputs, dim=0)

    return all_outputs


if __name__ == '__main__':
    root = 'C:/Users/Pc/Desktop/SilentTalk/'
    trained_on = 'asl100'
    checkpoint = 'ckpt.pth'
    video_file = '01.mp4'  # Replace with the path to your video file

    pose_data_root = os.path.join(root, 'data/pose_per_individual_videos')
    config_file = os.path.join(root, 'code/TGCN/archived/{}/{}.ini'.format(trained_on, trained_on))

    configs = Config(config_file)
    num_samples = configs.num_samples
    hidden_size = configs.hidden_size
    drop_p = configs.drop_p
    num_stages = configs.num_stages

    # Create a dummy dataset with the new video
    new_video_dataset = Sign_Dataset(video_file, split='test', pose_root=pose_data_root,
                                     img_transforms=None, video_transforms=None,
                                     num_samples=num_samples,
                                     sample_strategy='k_copies'
                                     )

    data_loader = DataLoader(dataset=new_video_dataset, batch_size=1, shuffle=False)

    # Setup the model
    model = GCN_muti_att(input_feature=num_samples * 2, hidden_feature=hidden_size,
                         num_class=int(trained_on[3:]), p_dropout=drop_p, num_stage=num_stages).cuda()

    # Load the trained model checkpoint
    print('Loading model...')
    checkpoint = torch.load(os.path.join(root, 'code/TGCN/archived/{}/{}'.format(trained_on, checkpoint)))
    model.load_state_dict(checkpoint)
    print('Model loaded successfully!')

    # Test the model on the new video
    outputs = test(model, data_loader)

    # Process or analyze the model's outputs as per your requirements
    print(outputs)