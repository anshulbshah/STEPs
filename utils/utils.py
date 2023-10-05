import pickle
from collections import deque
import torch
import random
import numpy as np 
from tqdm import tqdm

def set_random_seed(seed):
    # Seed all libraries
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def seed_worker(worker_id):
    # seed worker for sampling
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def sel_from_list(lst,indices):
    # select a subset of a list given indices
    new_list = []
    for idx in indices:
        new_list.append(lst[idx])
    return new_list

def extract_features_and_add_to_km(args,loader,model,km):
    # iterate a loader, pass sampled frames through the model and add to an object for later evaluation
    all_features = {}
    all_mid_frames = {}
    all_labels = {}
    all_names = {}
    all_subsampled_segment_list = {}
    for ii,feat_dict in tqdm(enumerate(loader),total=len(loader)):
        with torch.no_grad():
            input_global = feat_dict['clip_global'].cuda()
            B = feat_dict['clip_global'].shape[0]
            inputs_pose = {}
            for mod in args.modalities_to_use.split(',')[1:]:
                inputs_pose[mod] = feat_dict[mod].cuda()
            threshold_max_length = 75000
            T = input_global[0].shape[1]
            num_chunks_to_use = np.ceil(T/threshold_max_length).astype(int)
            features_all = []
            model_to_use = model
            assert num_chunks_to_use == 1
            input_to_use = (input_global,inputs_pose)
            _, _, features, projected_features  =  model_to_use(input_to_use,return_seq_features=True, timestamps=feat_dict['timestamps'])
            feature_to_use = features

            features_all.append(feature_to_use)
            features_all = torch.cat(features_all,1)
            mid_label = feat_dict['labels']
            for b in range(B):
                vid_idx = feat_dict['vid_idx'][b].item()
                all_features[vid_idx] = features_all[b].cpu()
                all_mid_frames[vid_idx] = list(feat_dict['timestamps'][b].numpy())
                all_labels[vid_idx] = list(mid_label[b].numpy())
                all_names[vid_idx] = feat_dict['video_path'][b]
                if 'subsampled_segment_list' in feat_dict.keys():
                    all_subsampled_segment_list[vid_idx] = feat_dict['subsampled_segment_list'][b]
                else:
                    all_subsampled_segment_list[vid_idx] = torch.arange(len(all_labels[vid_idx])).unsqueeze(0)

    for k in all_features.keys():
        if isinstance(all_features[k],list) or isinstance(all_features[k],tuple) :
            all_features[k] = torch.stack(all_features[k])
        all_mid_frames[k] = torch.Tensor(all_mid_frames[k])
        all_labels[k] = torch.Tensor(all_labels[k]).to(torch.int64)
        km.add_dataset(all_features[k].unsqueeze(0).detach().cpu(),\
                       all_mid_frames[k].unsqueeze(0).detach().cpu(),\
                       all_labels[k].unsqueeze(0).detach().cpu(),\
                       all_names[k],all_subsampled_segment_list[k].detach().cpu())
    print('Done collecting features!')

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name='null', fmt=':.4f'):
        self.name = name 
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.local_history = deque([])
        self.local_avg = 0
        self.history = []
        self.dict = {} # save all data values here
        self.save_dict = {} # save mean and std here, for summary table

    def update(self, val, n=1, history=0, step=5):
        self.val = val
        self.sum += val * n
        self.count += n
        if n == 0: return
        self.avg = self.sum / self.count
        if history:
            self.history.append(val)
        if step > 0:
            self.local_history.append(val)
            if len(self.local_history) > step:
                self.local_history.popleft()
            self.local_avg = np.average(self.local_history)


    def dict_update(self, val, key):
        if key in self.dict.keys():
            self.dict[key].append(val)
        else:
            self.dict[key] = [val]

    def print_dict(self, title='IoU', save_data=False):
        """Print summary, clear self.dict and save mean+std in self.save_dict"""
        total = []
        for key in self.dict.keys():
            val = self.dict[key]
            avg_val = np.average(val)
            len_val = len(val)
            std_val = np.std(val)

            if key in self.save_dict.keys():
                self.save_dict[key].append([avg_val, std_val])
            else:
                self.save_dict[key] = [[avg_val, std_val]]

            print('Activity:%s, mean %s is %0.4f, std %s is %0.4f, length of data is %d' \
                % (key, title, avg_val, title, std_val, len_val))

            total.extend(val)

        self.dict = {}
        avg_total = np.average(total)
        len_total = len(total)
        std_total = np.std(total)
        print('\nOverall: mean %s is %0.4f, std %s is %0.4f, length of data is %d \n' \
            % (title, avg_total, title, std_total, len_total))

        if save_data:
            print('Save %s pickle file' % title)
            with open('img/%s.pickle' % title, 'wb') as f:
                pickle.dump(self.save_dict, f)

    def __len__(self):
        return self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

class ProgressMeter(object):
    # Display a progress meter
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

if __name__ == '__main__':
    pass
