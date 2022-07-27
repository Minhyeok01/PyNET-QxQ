import argparse

parser = argparse.ArgumentParser(description='QxQ Demosaic Deep Learning Inference Model')


# Hardware specifications
parser.add_argument('--n_threads', type=int, default=1,
                    help='number of threads for data loading')
parser.add_argument('--cpu', action='store_true',
                    help='use cpu only')
parser.add_argument('--n_GPUs', type=int, default=1,
                    help='number of GPUs')
parser.add_argument('--GPU_id', type=str, default="0", choices=['0',"1",'2','3'],
                    help='if n_GPUs==1, specify GPU index')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed')


# Data specifications
parser.add_argument('--dir_data', type=str, default='./raw_images',
                    help='dataset directory')
parser.add_argument('--input_folder', type=str, default='msc',
                    help='input_folder')
parser.add_argument('--output_folder', type=str, default='gt',
                    help='output_folder')
parser.add_argument('--file_type', type=str, default='png',
                    help='file_type')
parser.add_argument('--output_file_type', type=str, default='png',
                    help='output_file_type')
parser.add_argument('--metric', action='store_true',
                    help='use validation metrics')
parser.add_argument('--data_test', type=str, default='DIV2K', choices=['DIV2K','Set5','Set14'],
                    help='test dataset name')


# Model specifications
parser.add_argument('--model', default='pynet_qxq_student_only',
                    help='model name')
parser.add_argument('--tag', default='0.1',
                    help='model tag')
parser.add_argument('--resume', type=int, default=0,
                    help='resume from specific checkpoint')
parser.add_argument('--cfa_type', type=str, default="QxQ",
                    help='cfa type')
parser.add_argument('--ratio_augmentation', type=int, default=0.0,
                    help='ratio_augmentation')
parser.add_argument('--inferencer', type=str, default='pynet_qxq_inference_eval',
                    help='custom inferencer')
parser.add_argument('--instance_norm_level_1', type=str, default='True',
                    help='set normalization')

# Log specifications
parser.add_argument('--export_root', type=str, default='./Experiments',
                    help='directory to export log file')

args = parser.parse_args()

for arg in vars(args):
    if vars(args)[arg] == 'True':
        vars(args)[arg] = True
    elif vars(args)[arg] == 'False':
        vars(args)[arg] = False
