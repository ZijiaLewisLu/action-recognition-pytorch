import os
import time

import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from tqdm import tqdm
import numpy as np

from models import build_model
# from utils.utils import build_dataflow, AverageMeter, accuracy
from utils.video_transforms import *
from utils.video_dataset import VideoRecord
from opts import arg_parser


def eval_a_batch(data, model, num_clips=1, num_crops=1, threed_data=False):
    with torch.no_grad():
        batch_size = data.shape[0]
        if threed_data:
            tmp = torch.chunk(data, num_clips * num_crops, dim=2)
            data = torch.cat(tmp, dim=0)
        else:
            data = data.view((batch_size * num_crops * num_clips, -1) + data.size()[2:])
        result = model(data)

        if threed_data:
            tmp = torch.chunk(result, num_clips * num_crops, dim=0)
            result = None
            for i in range(len(tmp)):
                result = result + tmp[i] if result is not None else tmp[i]
            result /= (num_clips * num_crops)
        else:
            result = result.reshape(batch_size, num_crops * num_clips, -1).mean(dim=1)

    return result

class VideoDataSet(torch.utils.data.Dataset):

    def __init__(self, root_path, list_file, image_tmpl='{:05d}.jpg', transform=None, seperator=' '):
        """
        Args:
            root_path (str): the file path to the root of video folder
            list_file (str): the file list, each line with folder_path, start_frame, end_frame, label_id
            image_tmpl (str): template of image ids
            transform: the transformer for preprocessing
        """

        self.root_path = root_path
        self.list_file = list_file
        self.image_tmpl = image_tmpl
        self.transform = transform
        self.seperator = seperator

        self.video_list = self._parse_list()

    def _image_path(self, directory, idx):
        return os.path.join(self.root_path, directory, self.image_tmpl.format(idx))

    def _load_image(self, directory, idx):

        def _safe_load_image(img_path):
            img_tmp = Image.open(img_path)
            img = img_tmp.copy()
            img_tmp.close()
            return img

        num_try = 0
        image_path_file = self._image_path(directory, idx)
        img = None
        while num_try < 10:
            try:
                img = [_safe_load_image(image_path_file)]
                break
            except Exception as e:
                print('[Will try load again] error loading image: {}, error: {}'.format(image_path_file, str(e)))
                num_try += 1

        if img is None:
            raise ValueError('[Fail 10 times] error loading image: {}'.format(image_path_file))

        return img

    def _parse_list(self):
        tmp = []
        original_video_numbers = 0
        for x in open(self.list_file):
            elements = x.strip().split(self.seperator)
            # start_frame = int(elements[1])
            # end_frame = int(elements[2])
            # total_frame = end_frame - start_frame + 1
            original_video_numbers += 1
            tmp.append(elements)

        num = len(tmp)
        print("The number of videos is {}".format(num), flush=True)
        assert (num > 0)
        file_list = []
        for item in tmp:
            file_list.append([item[0], int(item[1]), int(item[2]), -1])

        # VideoRecord( path, startframe, endframe, label )
        video_list = [VideoRecord(item[0], item[1], item[2], item[3]) for item in file_list]

        return video_list

    def __getitem__(self, index):
        """
        Returns:
            torch.FloatTensor: TxCxHxW dimension 
        """
        record = self.video_list[index]

        # load in all images
        images = [] 
        for i in range(record.start_frame, record.end_frame): 
            idx = i+1
            img = self._load_image(record.path, idx) # PIL image
            images.extend(img)

        images = self.transform(images) # torch tensor
        TC, H, W = images.shape
        images = images.view(-1, 3, H, W) # T, C, H, W

        return images

    def __len__(self):
        return len(self.video_list)

class BatchGenerator():

    def __init__(self, batch_size, video_tensor, group_size, frames_per_group):
        self.video = video_tensor
        self.group_size = group_size
        self.frames_per_group = frames_per_group
        self.window_size = group_size * frames_per_group
        self.batch_size = batch_size

        T = self.video.shape[0]
        self.num_window = max(1, 1 + T - self.window_size) # - 2 * 32 (num_frames = frames per group)
        self.batch_size = min(self.num_window, self.batch_size)
        self.batch_index = 0

    def _generate_window(self, start_index):
        frame_idx = np.arange(self.group_size) * self.frames_per_group + start_index 
        frames = self.video[frame_idx]
        return frames
    
    def __next__(self):
        if self.batch_index < self.num_window:
            batch = []
            for i in range(self.batch_size):
                batch.append(self._generate_window(self.batch_index))
                self.batch_index += 1
                if self.batch_index >= self.num_window:
                    break
            batch = torch.stack(batch, dim=0) # B, window_size, C, H, W
            batch = batch.permute(0, 2, 1, 3, 4) # B, C, window_size, H, W
            return batch
        else:
            raise StopIteration

    def __iter__(self):
        self.batch_index = 0
        return self

def main():
    global args
    parser = arg_parser()
    parser.add_argument('--feature_savedir', default=None, type=str, required=True)
    parser.add_argument('--video_list', default=None, type=str, required=True)
    args = parser.parse_args()
    cudnn.benchmark = True

    args.num_classes = num_classes = 400


    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    assert args.modality == 'rgb'
    assert args.num_crops == 1

    if args.modality == 'rgb':
        args.input_channels = 3
    elif args.modality == 'flow':
        args.input_channels = 2 * 5

    #############################
    # create model
    model, arch_name = build_model(args, test_mode=True)

    model = model.cuda()
    model.eval()

    if args.pretrained is not None:
        print("=> using pre-trained model '{}'".format(arch_name))
        checkpoint = torch.load(args.pretrained, map_location='cpu')
        model.load_state_dict(checkpoint['state_dict'])
    else:
        print("=> creating model '{}'".format(arch_name))


    ###########################
    # load dataset
    mean = model.mean(args.modality)
    std = model.std(args.modality)
    # overwrite mean and std if they are presented in command
    if args.mean is not None:
        if args.modality == 'rgb':
            if len(args.mean) != 3:
                raise ValueError("When training with rgb, dim of mean must be three.")
        elif args.modality == 'flow':
            if len(args.mean) != 1:
                raise ValueError("When training with flow, dim of mean must be three.")
        mean = args.mean

    if args.std is not None:
        if args.modality == 'rgb':
            if len(args.std) != 3:
                raise ValueError("When training with rgb, dim of std must be three.")
        elif args.modality == 'flow':
            if len(args.std) != 1:
                raise ValueError("When training with flow, dim of std must be three.")
        std = args.std


    # augmentor
    if args.disable_scaleup:
        scale_size = args.input_size
    else:
        scale_size = int(args.input_size / 0.875 + 0.5)

    augments = [
        GroupScale(scale_size),
        GroupCenterCrop(args.input_size),
        Stack(threed_data=args.threed_data),
        ToTorchFormatTensor(num_clips_crops=args.num_clips * args.num_crops), # second option is useless
        GroupNormalize(mean=mean, std=std, threed_data=args.threed_data)
    ]

    augmentor = transforms.Compose(augments)

    # Data loading code
    filename_seperator = ','
    image_tmpl = '{:03d}.jpg'
    # data_list_name = '/mnt/raptor/zijia/PTG/dataset/Breakfast/video_list_for_r3d_feature_extraction.txt'
    # target_dir = './MC2_feature'
    val_dataset = VideoDataSet(args.datadir, args.video_list, 
                                 image_tmpl=image_tmpl, 
                                 transform=augmentor, 
                                 seperator=filename_seperator, )


    ###################
    # start extracting feature
    V = len(val_dataset.video_list) 
    IDX = list(range(V))
    with torch.no_grad(), tqdm(total=len(IDX)) as t_bar:
        for i in IDX:
            vname = val_dataset.video_list[i].video_id

            save = os.path.join(args.feature_savedir, vname + '.npy')
            if os.path.exists(save):
                t_bar.update(1)
                continue

            video = val_dataset[i] # T, C, H, W

            # pad the beginning and end of the video
            slide_window = args.frames_per_group * args.groups
            half_window = np.ceil( (slide_window - 1) / 2 )
            half_window = int(half_window)

            head = video[0:1].repeat(half_window, 1, 1, 1)
            tail = video[-2:-1].repeat(half_window, 1, 1, 1)
            video = torch.cat([head, video, tail])

            # batch
            batch_iter = BatchGenerator(args.batch_size, video, args.groups, args.frames_per_group)
            
            with torch.no_grad():
                all_logits = []
                all_features = []
                for batch in batch_iter:
                    batch = batch.cuda() #.contiguous()
                    logits, features = model(batch)
                    # all_logits.append(logits.cpu())
                    features = features.mean(1) # B, window_size, 2048 -> B, 2048
                    all_features.append(features.detach().cpu())
                # all_logits = torch.cat(all_logits, dim=0)
                all_features = torch.cat(all_features, dim=0) # T, 2048
                all_features = all_features.numpy()

            all_features = np.save(save, all_features)

            t_bar.update(1)



if __name__ == '__main__':
    main()
