import os

from configs import Config
from sign_dataset import Sign_Dataset
import numpy as np
import torch
from sklearn.metrics import accuracy_score
import utils

from tgcn_model import GCN_muti_att


def create_label_mapping(dataset):
    # Create a mapping from video ID to gloss category
    label_mapping = {}

    for item in dataset.data:
        video_id = item['video_id']
        gloss_cat = item['gloss_cat']
        gloss = item['gloss']
        # Assuming gloss_cat is the categorical representation
        # original_gloss = utils.cat2labels.inverse_transform([gloss_cat])[0]

        # print(original_gloss)
        label_mapping[gloss_cat] = gloss

    return label_mapping



def test(model, test_loader, label_mapping):
    # print(label_mapping)
    # set model as testing mode
    model.eval()

    val_loss = []
    all_y = []
    all_y_pred = []
    all_video_ids = []
    all_pool_out = []
    all_text_predictions = []  # Store text predictions

    num_copies = 4

    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):
            print('starting batch: {}'.format(batch_idx))
            # distribute data to device
            X, y, video_ids,gloss = data
            X, y = X.cuda(), y.cuda().view(-1, ).long()

            all_output = []

            stride = X.size()[2] // num_copies

            for i in range(num_copies):
                X_slice = X[:, :, i * stride: (i + 1) * stride]
                output = model(X_slice)
                all_output.append(output)

            all_output = torch.stack(all_output, dim=1)
            output = torch.mean(all_output, dim=1)

            y_pred = output.max(1, keepdim=True)[1]  # (y_pred != output) get the index of the max log-probability

            # collect all y and y_pred in all batches
            all_y.extend(y)
            all_y_pred.extend(y_pred)
            all_video_ids.extend(video_ids)
            all_pool_out.extend(output)

            # print("predict",y_pred)

            # Convert numeric predictions to text labels
            text_predictions = [label_mapping[pred.item()] for pred in y_pred]
            all_text_predictions.extend(text_predictions)

    # Convert lists to numpy arrays
    all_y = torch.stack(all_y, dim=0).cpu()
    all_y_pred = torch.stack(all_y_pred, dim=0).cpu().squeeze()
    all_pool_out = torch.stack(all_pool_out, dim=0).cpu().data.numpy()

    # Calculate accuracy and other metrics
    top1acc = accuracy_score(all_y, all_y_pred)
    top3acc = compute_top_n_accuracy(all_y, all_pool_out, 3)
    top5acc = compute_top_n_accuracy(all_y, all_pool_out, 5)
    top10acc = compute_top_n_accuracy(all_y, all_pool_out, 10)
    top30acc = compute_top_n_accuracy(all_y, all_pool_out, 30)

    # Output metrics
    print('\nVal. set ({:d} samples): top-1 Accuracy: {:.2f}%\n'.format(len(all_y), 100 * top1acc))
    print('\nVal. set ({:d} samples): top-3 Accuracy: {:.2f}%\n'.format(len(all_y), 100 * top3acc))
    print('\nVal. set ({:d} samples): top-5 Accuracy: {:.2f}%\n'.format(len(all_y), 100 * top5acc))
    print('\nVal. set ({:d} samples): top-10 Accuracy: {:.2f}%\n'.format(len(all_y), 100 * top10acc))

    # Output text predictions
    print('Text Predictions:', all_text_predictions)

        # Return text predictions
    return all_text_predictions


# Helper function to compute top-n accuracy
def compute_top_n_accuracy(truths, preds, n):
    best_n = np.argsort(preds, axis=1)[:, -n:]
    ts = truths
    successes = 0
    for i in range(ts.shape[0]):
        if ts[i] in best_n[i, :]:
            successes += 1
    return float(successes) / ts.shape[0]

if __name__ == '__main__':
    # change root and subset accordingly.
    root = 'C:/Users/Pc/Desktop/SilentTalk/'
    trained_on = 'asl100'

    checkpoint = 'ckpt.pth'

    split_file = os.path.join(root, 'data/splits/{}.json'.format(trained_on))
    # test_on_split_file = os.path.join(root, 'data/splits-with-dialect-annotated/{}.json'.format(tested_on))

    pose_data_root = os.path.join(root, 'data/pose_per_individual_videos')
    # config_file = os.path.join(root, 'code/TGCN/archived/asl100/asl100.ini')

    config_file = os.path.join(root, 'code/TGCN/archived/{}/{}.ini'.format(trained_on, trained_on))
    print(config_file)
    configs = Config(config_file)

    num_samples = configs.num_samples
    hidden_size = configs.hidden_size
    drop_p = configs.drop_p
    num_stages = configs.num_stages
    batch_size = configs.batch_size

    dataset = Sign_Dataset(index_file_path=split_file, split='test', pose_root=pose_data_root,
                           img_transforms=None, video_transforms=None,
                           num_samples=num_samples,
                           sample_strategy='k_copies',
                           test_index_file=split_file
                           )
    
    for item in dataset:
        print(item)
    

    data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

    # setup the model
    model = GCN_muti_att(input_feature=num_samples * 2, hidden_feature=hidden_size,
                         num_class=int(trained_on[3:]), p_dropout=drop_p, num_stage=num_stages).cuda()

    print('Loading model...')

    checkpoint = torch.load(os.path.join(root, 'code/TGCN/archived/{}/{}'.format(trained_on, checkpoint)))
    model.load_state_dict(checkpoint)
    print('Finish loading model!')
    
    # Create label mapping
    label_mapping = create_label_mapping(dataset)

    # Test the model
    test(model, data_loader, label_mapping)
