import sys
sys.path.append("..")

import os
import time
import json
import logging
import argparse

import torch
import torch.backends.cudnn as cudnn

import dataset
from train.model import static_model
from train import metric
from data import video_sampler as sampler
from data import video_transforms as transforms
from data.video_iterator import VideoIter
from network.symbol_builder import get_symbol
from mlp import MLP, MLP10, MLP512, MLP1024
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names

return_nodes = {
    'resnet.layer1': 'resnet.layer1',
    'resnet.layer2': 'resnet.layer2',
    'resnet.layer3': 'resnet.layer3',
    'resnet.layer4': 'resnet.layer4',
    'resnet.avgpool': 'resnet.avgpool',
}

torch.backends.cudnn.enabled = False

parser = argparse.ArgumentParser(description="PyTorch Video Recognition Parser (Evaluation) default UCF101")
# debug
parser.add_argument('--debug-mode', type=bool, default=True,
                    help="print all setting for debugging.")
# io
parser.add_argument('--dataset', default='ARID', help="path to dataset")
parser.add_argument('--clip-length', default=16,
                    help="define the length of each input sample.")    
parser.add_argument('--frame-interval', type=int, default=2,
                    help="define the sampling interval between frames.")    
# parser.add_argument('--task-name', type=str, default='../exps/models/archive/ARID_PyTorch',
# 					help="name of current task, leave it empty for using folder name")
parser.add_argument('--model-dir', type=str, default="./",
                    help="set logging file.")
parser.add_argument('--log-file', type=str, default="../exps/logs_test",
                    help="set logging file.")
# device
parser.add_argument('--gpus', type=str, default="0",
                    help="define gpu id")
# algorithm
parser.add_argument('--network', type=str, default='resnext',
                    help="choose the base network")
# evaluation
parser.add_argument('--load-epoch', type=int, default=50,
                    help="resume trained model")
parser.add_argument('--batch-size', type=int, default=8,
                    help="batch size")

#other changes
parser.add_argument('--list-file', type=str, default='ARID_split1_test.txt',
                    help='list of testing videos, see list_cvt folder of each dataset for details')
parser.add_argument('--workers', type=int, default=4, help='num_workers during evaluation data loading')
parser.add_argument('--test-rounds', type=int, default=0, help='number of testing rounds')
parser.add_argument('--is-dark', type=bool, default=False)
parser.add_argument('--brighten-method', type=str, 
                    default=None,
                    help="lime/ying_caip")
parser.add_argument('--brighten-method2', type=str, 
                    default=None,
                    help="lime/ying_caip")
parser.add_argument('--task-name', type=str, default='',
                    help="name of current task, leave it empty for using folder name")
parser.add_argument('--model-path', type=str, default='',
                    help="directly point to the .pth file")
parser.add_argument('--fusion-mode', type=str, default=None,
                    help="none/loose/tight")
parser.add_argument('--model2-path', type=str, default='',
                    help="directly point to the .pth file of the canny network")
parser.add_argument('--mode', type=str, default='val', help='train/val/test')
parser.add_argument('--save-result', action='store_true', default=False, help='whether to save predictions')


def autofill(args):
    # customized
    if not args.task_name:
        args.task_name = os.path.basename(os.getcwd())
    # fixed
    args.model_prefix = os.path.join(args.model_dir, args.task_name)
    return args

def set_logger(log_file='', debug_mode=False):
    if log_file:
        if not os.path.exists("./"+os.path.dirname(log_file)):
            os.makedirs("./"+os.path.dirname(log_file))
        handlers = [logging.FileHandler(log_file), logging.StreamHandler()]
    else:
        handlers = [logging.StreamHandler()]

    """ add '%(filename)s' to format show source file """
    logging.basicConfig(level=logging.DEBUG if debug_mode else logging.INFO,
                format='%(asctime)s %(levelname)s: %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S',
                handlers = handlers)

def main(args):
    set_logger(log_file=os.path.join(args.log_file, 'test_%s' % args.task_name), debug_mode=args.debug_mode)
    logging.info("Start evaluation with args:\n" +
                 json.dumps(vars(args), indent=4, sort_keys=True))

    # set device states
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus # before using torch
    assert torch.cuda.is_available(), "CUDA is not available"

    # load dataset related configuration
    dataset_cfg = dataset.get_config(name=args.dataset)

    # creat model
    dark = args.is_dark
    sym_net, input_config = get_symbol(name=args.network, is_dark=dark, **dataset_cfg)
    
    # network
    if torch.cuda.is_available():
        cudnn.benchmark = True
        sym_net = torch.nn.DataParallel(sym_net).to('cuda:%s' % device)
        criterion = torch.nn.CrossEntropyLoss().to('cuda:%s' % device)
    else:
        sym_net = torch.nn.DataParallel(sym_net)
        criterion = torch.nn.CrossEntropyLoss()
    net = static_model(net=sym_net,
                       criterion=criterion,
                       model_prefix=args.model_prefix)
    net.load_checkpoint(epoch=9999, model_path=args.model_path)
    
    # data iterator:
    data_root = "../dataset/{}".format(args.dataset)
    
    if args.dataset.upper() == 'MINIKINETICS':
        video_location = os.path.join(data_root, 'raw', 'data', 'val')
    # elif args.test:
    #     video_location = os.path.join(data_root, 'test_%s' % args.brighten_method if args.brighten_method is not None else 'test' )
    else:
        if args.brighten_method is not None:
            video_location = os.path.join(data_root, '%s_%s' % (args.mode, args.brighten_method) )
        else:
            video_location = os.path.join(data_root, '%s' % (args.mode) )

            
    normalize = transforms.Normalize(mean=input_config['mean'], std=input_config['std'])
    # val_sampler = sampler.RandomSampling(num=args.clip_length,
    # 									 interval=args.frame_interval,
    # 									 speed=[1.0, 1.0], seed=1)
    val_sampler = sampler.MGSampler(num=args.clip_length, mode='val')


    val_loader = VideoIter(video_prefix=video_location,
                      txt_list=os.path.join(data_root, '%s.txt' % args.mode), 
                      sampler=val_sampler,
                      force_color=True,
                    #   video_transform=transforms.Compose([
                    # 					 transforms.Resize((256,256)),
                    # 					 transforms.RandomCrop((224,224)),
                    # 					 # transforms.Resize((128, 128)), # For C3D Only
                    # 					 # transforms.RandomCrop((112, 112)), # For C3D Only
                    # 					 # transforms.CenterCrop((224, 224)), # we did not use center crop in our paper
                    # 					 # transforms.RandomHorizontalFlip(), # we did not use mirror in our paper
                    # 					 transforms.ToTensor(),
                    # 					#  normalize,
                    # 				  ]),
                      video_transform=transforms.Compose([
                                         # transforms.Resize((128, 128)), # For C3D Only
                                         # transforms.CenterCrop((112, 112)), # For C3D Only
                                         # transforms.Resize((256, 256)),
                                           transforms.RandomScale(make_square=False,
                                            aspect_ratio=[1.0, 1.0],
                                            slen=[256, 256]),
                                         transforms.CenterCrop((224, 224)),
                                         transforms.ToTensor(),
                                         normalize,
                                      ]),
                      name='test',
                      return_item_subpath=True,
                      )
    assert args.batch_size == 1, 'not support more than 1 batch size evaluation'
    eval_iter = torch.utils.data.DataLoader(val_loader,
                      batch_size=args.batch_size,
                      shuffle=False,
                      num_workers=args.workers,
                      pin_memory=False)

    # main loop
    net.net.eval()
    softmax = torch.nn.Softmax(dim=1)
    correct = []
    output_list = []
    for data, target, _ in eval_iter:

        outputs, losses = net.forward(data, target)


        # recording
        output = softmax(outputs[0]).data.cpu()
        correct.append(output.topk(1,1)[1].item() == target.item())
        output_list.append(output.topk(1,1)[1].item())
        # print(outputs)
        # time.sleep(10000)
        target = target.cpu()
        losses = losses[0].data.cpu()
    if args.save_result:
        save_file = open('%s_output.txt' % args.mode, 'w')
        for id, i in enumerate(correct):
            print('%s.mp4\t%s' % (id, output_list[id]), file=save_file)
        save_file.close()

    logging.info("top 1 accuracy is %s" % (sum(correct) / len(correct)))

    # finished
    logging.info("Evaluation Finished!")


  

if __name__ == '__main__':
    # set args
    args = parser.parse_args()
    args = autofill(args)
    global device
    device = args.gpus
    main(args)