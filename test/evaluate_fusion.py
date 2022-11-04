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
    else:
        video_location = os.path.join(data_root, 'val_%s' % args.brighten_method if args.brighten_method is not None else 'val' )
    
    normalize = transforms.Normalize(mean=input_config['mean'], std=input_config['std'])
    # val_sampler = sampler.RandomSampling(num=args.clip_length,
    # 									 interval=args.frame_interval,
    # 									 speed=[1.0, 1.0], seed=1)
    val_sampler = sampler.MGSampler(num=args.clip_length, mode='val')


    val_loader = VideoIter(video_prefix=video_location,
                      txt_list=os.path.join(data_root, 'val.txt'), 
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
                      
    eval_iter = torch.utils.data.DataLoader(val_loader,
                      batch_size=args.batch_size,
                      shuffle=False,
                      num_workers=args.workers,
                      pin_memory=False)

    # eval metrics
    metrics = metric.MetricList(metric.Loss(name="loss-ce"),
                                metric.Accuracy(topk=1, name="top1"),
                                metric.Accuracy(topk=5, name="top5"))
    metrics.reset()

    # main loop
    net.net.eval()
    avg_score = {}
    sum_batch_elapse = 0.
    sum_batch_inst = 0
    duplication = 1
    softmax = torch.nn.Softmax(dim=1)

    total_round = args.test_rounds # change this part accordingly if you do not want an inf loop
    for i_round in range(total_round):
        i_batch = 0
        logging.info("round #{}/{}".format(i_round, total_round))
        for data, target, video_subpath in eval_iter:
            batch_start_time = time.time()

            outputs, losses = net.forward(data, target)

            sum_batch_elapse += time.time() - batch_start_time
            sum_batch_inst += 1

            # recording
            output = softmax(outputs[0]).data.cpu()
            # print(outputs)
            # time.sleep(10000)
            target = target.cpu()
            losses = losses[0].data.cpu()
            # logging.info("output is {}, target is {}".format(output, target))
            for i_item in range(0, output.shape[0]):
                output_i = output[i_item,:].view(1, -1)
                target_i = torch.LongTensor([target[i_item]])
                loss_i = losses
                video_subpath_i = video_subpath[i_item]
                if video_subpath_i in avg_score:
                    avg_score[video_subpath_i][2] += output_i
                    avg_score[video_subpath_i][3] += 1
                    duplication = 0.92 * duplication + 0.08 * avg_score[video_subpath_i][3]
                else:
                    avg_score[video_subpath_i] = [torch.LongTensor(target_i.numpy().copy()), 
                                                  torch.FloatTensor(loss_i.numpy().copy()), 
                                                  torch.FloatTensor(output_i.numpy().copy()),
                                                  1] # the last one is counter

            # show progress
            if (i_batch % (len(eval_iter) - 1)) == 0:
                metrics.reset()
                for _, video_info in avg_score.items():
                    target, loss, pred, _ = video_info
                    metrics.update([pred], target, [loss])
                name_value = metrics.get_name_value()
                logging.info("{:.1f}%, {:.1f} \t| Batch [0,{}]    \tAvg: {} = {:.5f}, {} = {:.5f}, {} = {:.5f}".format(
                            float(100*i_batch) / eval_iter.__len__(), \
                            duplication, \
                            i_batch, \
                            name_value[0][0][0], name_value[0][0][1], \
                            name_value[1][0][0], name_value[1][0][1], \
                            name_value[2][0][0], name_value[2][0][1]))
            i_batch += 1


    # finished
    logging.info("Evaluation Finished!")

    metrics.reset()
    for _, video_info in avg_score.items():
        target, loss, pred, _ = video_info
        metrics.update([pred], target, [loss])

    logging.info("Total time cost: {:.1f} sec".format(sum_batch_elapse))
    logging.info("Speed: {:.4f} samples/sec".format(
            args.batch_size * sum_batch_inst / sum_batch_elapse ))
    logging.info("Accuracy:")
    logging.info(json.dumps(metrics.get_name_value(), indent=4, sort_keys=True))

def loose_fusion(args):
    set_logger(log_file=os.path.join(args.log_file, 'test_%s' % args.task_name), debug_mode=args.debug_mode)
    logging.info("Start evaluation with args:\n" +
                 json.dumps(vars(args), indent=4, sort_keys=True))

    # set device states
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus # before using torch
    # assert torch.cuda.is_available(), "CUDA is not available"

    # load dataset related configuration
    dataset_cfg = dataset.get_config(name=args.dataset)

    # creat model
    dark = args.is_dark
    sym_net, input_config = get_symbol(name=args.network, is_dark=dark, **dataset_cfg)
    sym_net2, input_config2 = get_symbol(name=args.network, is_dark=dark, **dataset_cfg)
    
    # network
    if torch.cuda.is_available():
        cudnn.benchmark = True
        sym_net = torch.nn.DataParallel(sym_net).to('cuda:%s' % device)
        criterion = torch.nn.CrossEntropyLoss().to('cuda:%s' % device)
        sym_net2 = torch.nn.DataParallel(sym_net2).to('cuda:%s' % device)
        criterion2 = torch.nn.CrossEntropyLoss().to('cuda:%s' % device)
    else:
        sym_net = torch.nn.DataParallel(sym_net)
        criterion = torch.nn.CrossEntropyLoss()
    net = static_model(net=sym_net,
                       criterion=criterion,
                       model_prefix=args.model_prefix)
    net2 = static_model(net=sym_net2,
                       criterion=criterion2,
                       model_prefix=args.model_prefix)
    net.load_checkpoint(epoch=9999, model_path=args.model_path)
    net2.load_checkpoint(epoch=9999, model_path=args.model2_path)
    
    # data iterator:
    data_root = "../dataset/{}".format(args.dataset)
    
    if args.dataset.upper() == 'MINIKINETICS':
        video_location = os.path.join(data_root, 'raw', 'data', 'val')
    else:
        video_location = os.path.join(data_root, 'val_%s' % args.brighten_method if args.brighten_method is not None else 'val' )
        video_location2 = os.path.join(data_root, 'val_%s' % args.brighten_method2 if args.brighten_method2 is not None else 'val' )
    normalize = transforms.Normalize(mean=input_config['mean'], std=input_config['std'])
    # val_sampler = sampler.RandomSampling(num=args.clip_length,
    # 									 interval=args.frame_interval,
    # 									 speed=[1.0, 1.0], seed=1)
    val_sampler = sampler.MGSampler(num=args.clip_length, mode='val')


    val_loader = VideoIter(video_prefix=video_location,
                      txt_list=os.path.join(data_root, 'val.txt'), 
                      sampler=val_sampler,
                      force_color=True,
                      video_transform=transforms.Compose([
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
    val_loader2 = VideoIter(video_prefix=video_location2,
                      txt_list=os.path.join(data_root, 'val.txt'), 
                      sampler=val_sampler,
                      force_color=True,
                      video_transform=transforms.Compose([
                                           transforms.RandomScale(make_square=False,
                                            aspect_ratio=[1.0, 1.0],
                                            slen=[256, 256]),
                                         transforms.CenterCrop((224, 224)),
                                         transforms.ToTensor(),
                                      ]),
                      name='test',
                      return_item_subpath=True,
                      )
                      
    eval_iter = torch.utils.data.DataLoader(val_loader,
                      batch_size=args.batch_size,
                      shuffle=False,
                      num_workers=args.workers,
                      pin_memory=False)
    eval_iter2 = torch.utils.data.DataLoader(val_loader2,
                      batch_size=args.batch_size,
                      shuffle=False,
                      num_workers=args.workers,
                      pin_memory=False)

    # eval metrics
    metrics = metric.MetricList(metric.Loss(name="loss-ce"),
                                metric.Accuracy(topk=1, name="top1"),
                                metric.Accuracy(topk=5, name="top5"))
    metrics.reset()
    # main loop
    net.net.eval()
    net2.net.eval()
    avg_score = {}
    sum_batch_elapse = 0.
    sum_batch_inst = 0
    duplication = 1
    softmax = torch.nn.Softmax(dim=1)

    total_round = args.test_rounds # change this part accordingly if you do not want an inf loop
    for i_round in range(total_round):
        i_batch = 0
        fusion_correct = 0
        rgb_correct = 0
        canny_correct = 0
        f = open('val_result.txt', 'w')
        print('rgv\tcanny\tfusion\ttarget\n', file=f)
        logging.info("round #{}/{}".format(i_round, total_round))
        with torch.no_grad():
            for (data, target, video_subpath), (data2, target2, video_subpath2) in zip(eval_iter, eval_iter2):
                batch_start_time = time.time()

                outputs, losses = net.forward(data, target)
                outputs2, losses2 = net2.forward(data2, target2)
                print(outputs)
                print(outputs2)
                time.sleep(1000)
                sum_batch_elapse += time.time() - batch_start_time
                sum_batch_inst += 1

                # recording
                output = softmax(outputs[0]).data.cpu()
                output2 = softmax(outputs2[0]).data.cpu()
                # print(outputs)
                # time.sleep(10000)
                target = target.cpu()
                target2 = target2.cpu()
                output_fusion = 0.528*output + 0.494*output2
                if output.topk(1,1)[1].item() == target.item():
                    rgb_correct += 1
                if output2.topk(1,1)[1].item() == target.item():
                    canny_correct += 1
                if output_fusion.topk(1,1)[1].item() == target.item():
                    fusion_correct += 1
                print("%s\t%s\t%s\t%s\n" % (output.topk(1,1)[1].item(), output2.topk(1,1)[1].item(), output_fusion.topk(1,1)[1].item(), target.item()),file=f)
                losses = losses[0].data.cpu()
                losses2 = losses2[0].data.cpu()


        f.close()
        print("==========================================================================")
        print("fusion accuracy is %s" % (fusion_correct / len(eval_iter)))
        print("rgb accuracy is %s" % (rgb_correct / len(eval_iter)))
        print("canny accuracy is %s" % (canny_correct / len(eval_iter)))

    # finished
    logging.info("Evaluation Finished!")

def mlp_loose_fusion(args):
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
    sym_net2, input_config2 = get_symbol(name=args.network, is_dark=dark, **dataset_cfg)
    
    # network
    if torch.cuda.is_available():
        cudnn.benchmark = True
        # sym_net = torch.nn.DataParallel(sym_net).to('cuda:%s' % device)
        sym_net = sym_net.to('cuda:%s' % device)

        criterion = torch.nn.CrossEntropyLoss().to('cuda:%s' % device)
        # sym_net2 = torch.nn.DataParallel(sym_net2).to('cuda:%s' % device)
        sym_net2 = sym_net2.to('cuda:%s' % device)

        criterion2 = torch.nn.CrossEntropyLoss().to('cuda:%s' % device)
    else:
        sym_net = torch.nn.DataParallel(sym_net)
        criterion = torch.nn.CrossEntropyLoss()
    net = static_model(net=sym_net,
                       criterion=criterion,
                       model_prefix=args.model_prefix)
    net2 = static_model(net=sym_net2,
                       criterion=criterion2,
                       model_prefix=args.model_prefix)
    net.load_checkpoint(epoch=9999, model_path=args.model_path)
    net2.load_checkpoint(epoch=9999, model_path=args.model2_path)
    
    # data iterator for val:
    data_root = "../dataset/{}".format(args.dataset)
    
    if args.dataset.upper() == 'MINIKINETICS':
        video_location = os.path.join(data_root, 'raw', 'data', 'val')
    else:
        video_location = os.path.join(data_root, 'val_%s' % args.brighten_method if args.brighten_method is not None else 'val' )
        video_location2 = os.path.join(data_root, 'val_%s' % args.brighten_method2 if args.brighten_method2 is not None else 'val' )
    normalize = transforms.Normalize(mean=input_config['mean'], std=input_config['std'])
    val_sampler = sampler.MGSampler(num=args.clip_length, mode='val')


    val_loader = VideoIter(video_prefix=video_location,
                      txt_list=os.path.join(data_root, 'val.txt'), 
                      sampler=val_sampler,
                      force_color=True,
                      video_transform=transforms.Compose([
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
    val_loader2 = VideoIter(video_prefix=video_location2,
                      txt_list=os.path.join(data_root, 'val.txt'), 
                      sampler=val_sampler,
                      force_color=True,
                      video_transform=transforms.Compose([
                                           transforms.RandomScale(make_square=False,
                                            aspect_ratio=[1.0, 1.0],
                                            slen=[256, 256]),
                                         transforms.CenterCrop((224, 224)),
                                         transforms.ToTensor(),
                                      ]),
                      name='test',
                      return_item_subpath=True,
                      )
                      
    eval_iter = torch.utils.data.DataLoader(val_loader,
                      batch_size=args.batch_size,
                      shuffle=False,
                      num_workers=args.workers,
                      pin_memory=False)
    eval_iter2 = torch.utils.data.DataLoader(val_loader2,
                      batch_size=args.batch_size,
                      shuffle=False,
                      num_workers=args.workers,
                      pin_memory=False)

    # data iterator for train mlp:
    data_root = "../dataset/{}".format(args.dataset)
    
    if args.dataset.upper() == 'MINIKINETICS':
        video_location = os.path.join(data_root, 'raw', 'data', 'val')
    else:
        video_location_train = os.path.join(data_root, 'train_%s' % args.brighten_method if args.brighten_method is not None else 'train' )
        video_location_train2 = os.path.join(data_root, 'train_%s' % args.brighten_method2 if args.brighten_method2 is not None else 'train' )
    normalize = transforms.Normalize(mean=input_config['mean'], std=input_config['std'])
    train_sampler = sampler.MGSampler(num=args.clip_length, mode='train')


    train_loader = VideoIter(video_prefix=video_location_train,
                      txt_list=os.path.join(data_root, 'train.txt'), 
                      sampler=train_sampler,
                      force_color=True,
                      video_transform=transforms.Compose([
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
    train_loader2 = VideoIter(video_prefix=video_location_train2,
                      txt_list=os.path.join(data_root, 'train.txt'), 
                      sampler=val_sampler,
                      force_color=True,
                      video_transform=transforms.Compose([
                                           transforms.RandomScale(make_square=False,
                                            aspect_ratio=[1.0, 1.0],
                                            slen=[256, 256]),
                                         transforms.CenterCrop((224, 224)),
                                         transforms.ToTensor(),
                                      ]),
                      name='test',
                      return_item_subpath=True,
                      )
                      
    train_iter = torch.utils.data.DataLoader(train_loader,
                      batch_size=8,
                      shuffle=False,
                      num_workers=args.workers,
                      pin_memory=False)
    train_iter2 = torch.utils.data.DataLoader(train_loader2,
                      batch_size=8,
                      shuffle=False,
                      num_workers=args.workers,
                      pin_memory=False)


    # main loop for eval 
    net.net.eval()
    net2.net.eval()
 
    # train mlp to fusion the result of MODEL1 AND MODEL2
    # model = MLP10()
    # model = model.to('cuda:%s' % device)
    # lossfunc = torch.nn.CrossEntropyLoss()
    # optimizer = torch.optim.SGD(params=model.parameters(), lr=0.01)
    # softmax = torch.nn.Softmax(dim=1)
    # print(net2.net)
    # time.sleep(100000)
    # print(get_graph_node_names(net.net))
    # time.sleep(100000)
    feat_extract = create_feature_extractor(net.net, return_nodes=return_nodes)
    feat_extract = feat_extract.to('cuda:%s' % device)
    feat_extract2 = create_feature_extractor(net2.net, return_nodes=return_nodes)
    feat_extract2 = feat_extract.to('cuda:%s' % device)
    
    # # for normal mlp10
    # best_accuracy = 0.0
    # for epoch in range(100):

    #     model.train()
    #     train_loss = 0.0
    #     for (data, target, _), (data2, target2, __) in zip(train_iter, train_iter2):
    #         with torch.no_grad():
    #             outputs, losses = net.forward(data, target)
    #             outputs2, losses2 = net2.forward(data2, target2)
    #             output_corr = outputs * outputs2
    #             output_corr = (output_corr - torch.mean(output_corr, 1).reshape(-1, 1)) - torch.std(output_corr, 1).reshape(-1, 1)
    #         optimizer.zero_grad()
    #         output_corr = output_corr.to('cuda:%s' % device)
    #         output_final = model(output_corr)	
    #         target = target.to('cuda:%s' % device)
    #         loss = lossfunc(output_final, target)
    #         loss.backward()
    #         optimizer.step()
    #         train_loss += loss.item()*output_corr.size(0)
    #     fusion_correct = 0
    #     for (data_val, target_val, video_subpath), (data_val2, target_val2, video_subpath2) in zip(eval_iter, eval_iter2):
    #         batch_start_time = time.time()
    #         with torch.no_grad():
    #             outputs_val, losses = net.forward(data_val, target_val)
    #             outputs_val2, losses2 = net2.forward(data_val2, target_val2)
    #             output_corr_val = outputs_val * outputs_val2
    #             output_corr_val = (output_corr_val - torch.mean(output_corr_val, 1).reshape(-1, 1)) - torch.std(output_corr_val, 1).reshape(-1, 1)
    #         model.eval()
    #         output_corr_val = output_corr_val.to('cuda:%s' % device)
    #         output_final_val = model(output_corr_val)
    #         # recording
    #         output_final_val = softmax(output_final_val).data.cpu()

    #         target_val = target_val.cpu()
    #         if output_final_val.topk(1,1)[1].item() == target_val.item():
    #             fusion_correct += 1
    #     fusion_accuracy = fusion_correct / len(eval_iter)
    #     if fusion_accuracy > best_accuracy:
    #         best_accuracy = fusion_accuracy
    #         print("current best model is epoch %s and accuracy is %s" % (epoch, best_accuracy))
    #         torch.save(model, "mlp_10/best.pth")
    
    
    model = MLP1024()
    model = model.to('cuda:%s' % device)
    lossfunc = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(params=model.parameters(), lr=0.01)
    softmax = torch.nn.Softmax(dim=1)
    # # for fusion features 
    best_accuracy = 0.0
    for epoch in range(100):
        
        model.train()
        train_loss = 0.0
        for (data, target, _), (data2, target2, __) in zip(train_iter, train_iter2):
            data = data.to('cuda:%s' % device)
            data2 = data2.to('cuda:%s' % device)
            feature = feat_extract(data)['resnet.avgpool'].squeeze(-1).squeeze(-1).squeeze(-1)
            feature2 = feat_extract2(data2)['resnet.avgpool'].squeeze(-1).squeeze(-1).squeeze(-1)
            # output_corr = feature * feature2
            output_corr = torch.concat((feature, feature2), dim=1)
            # output_corr = (output_corr - torch.mean(output_corr, 1).reshape(-1,1)) / (torch.std(output_corr, 1).reshape(-1, 1) + 1e-6)
            
            optimizer.zero_grad()
            output_corr = output_corr.to('cuda:%s' % device)
            output_final = model(output_corr)	
            target = target.to('cuda:%s' % device)

            loss = lossfunc(output_final, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()*output_corr.size(0)
        fusion_correct = 0
        for (data_val, target_val, video_subpath), (data_val2, target_val2, video_subpath2) in zip(eval_iter, eval_iter2):
            data_val = data_val.to('cuda:%s' % device)
            data_val2 = data_val2.to('cuda:%s' % device)
            with torch.no_grad():
                feature = feat_extract(data_val)['resnet.avgpool'].squeeze(-1).squeeze(-1).squeeze(-1)
                feature2 = feat_extract2(data_val2)['resnet.avgpool'].squeeze(-1).squeeze(-1).squeeze(-1)
                # output_corr_val = feature * feature2
                output_corr_val = torch.concat((feature, feature2), dim=1)
                # output_corr_val = (output_corr_val - torch.mean(output_corr_val, 1).reshape(-1,1)) / (torch.std(output_corr_val, 1).reshape(-1, 1) + 1e-6)
            model.eval()
            output_corr_val = output_corr_val.to('cuda:%s' % device)
            output_final_val = model(output_corr_val)
            # recording
            output_final_val = softmax(output_final_val).data.cpu()

            target_val = target_val.cpu()
            if output_final_val.topk(1,1)[1].item() == target_val.item():
                fusion_correct += 1
        fusion_accuracy = fusion_correct / len(eval_iter)
        if fusion_accuracy > best_accuracy:
            best_accuracy = fusion_accuracy
            print("current best model is epoch %s and accuracy is %s" % (epoch, best_accuracy))
            torch.save(model, "mlp_1024/best.pth")
        else:
            print("epoch %s accuracy is %s" % (epoch, fusion_accuracy))
            
  

if __name__ == '__main__':
    # set args
    args = parser.parse_args()
    args = autofill(args)
    global device
    device = args.gpus
    if args.fusion_mode is None:
        main(args)
    elif args.fusion_mode == 'loose':
        loose_fusion(args)
    elif args.fusion_mode == 'tight':
        tight_fusion(args)
    elif args.fusion_mode == 'train_mlp_loose':
        mlp_loose_fusion(args)
    else:
        print("error!!!!!!!!, no such fusion mode")
    
