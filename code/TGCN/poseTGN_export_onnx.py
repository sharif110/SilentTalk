import argparse
import torch
from tgcn_model import GCN_muti_att


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--weight', type=str, required=True)
    parser.add_argument("-o", '--output_name', type=str, required=True)
    args = parser.parse_args()

    num_classes =100

    net = GCN_muti_att(input_feature=50 * 2, hidden_feature=64,
                         num_class=num_classes, p_dropout=0.3, num_stage=20)

    net.load_state_dict(torch.load(args.weight))

    input = torch.randn(1, 55, 50 * 2)
    input_names = ['data']
    output_names = ['output']

    torch.onnx.export(net, input, args.output_name,
                      verbose=True,
                      input_names=input_names,
                      output_names=output_names)