import os
import numpy as np
import torch
import pickle
import numpy as np
import time
from torch.utils.data import Dataset
import glob
from collections import defaultdict
from pathlib import Path
import pandas as pd

def get_rng(args):
    return np.random.default_rng(args.random_seed)


class MeccanoDataset(Dataset):
    # Custom dataset for loading Meccano features
    def __init__(self, args,set='test', rmode='train'):
        
        self.set = set
        self.rmode = rmode
        self.num_chunks = args.num_chunks
        self.num_chunks_val = args.num_chunks_val
        self.parsed_dataset_path = args.parsed_dataset_path
        self.args = args
        
        self.modalities_to_use = args.modalities_to_use
        self.fps = 12
        self.num_keysteps = 17

        if 'rgb_res50' in self.modalities_to_use:
            # load rgb resnet50 features (N x D1) and mapping from image to location in the feature file
            found_file_name = glob.glob(os.path.join(args.parsed_dataset_path,'features','resnet50_features',f'*conv5c*.dat'))
            feature_location_to_use = feature_location = found_file_name[0]
            _,_,num_vids,num_features = feature_location_to_use.split('/')[-1].split('.')[0].split('_')
            self.feature_mapping_og = pickle.load(open(feature_location_to_use.replace('.dat','.pkl'),'rb'))
            self.feature_mapping = {'/'.join(k.split('/')[1:]):v for k,v in self.feature_mapping_og.items()}
            assert len(self.feature_mapping) == int(num_vids)
            self.feature_data =  np.memmap(feature_location_to_use,dtype=np.float32,shape=(int(num_vids),int(num_features)),mode='r')
            # note that we have 2049 dimensional features. The last dimension is just a counter which we remove when using these features with the temporal encoder. 
            self.rgb_dim = int(num_features)-1

        if 'raft_motion_features_pose' in self.modalities_to_use:
            # load raft OF features (N x D2) and mapping from image to location in the feature file
            self.pose_data = np.load(f'{args.parsed_dataset_path}/features/raft_motion_features/raft_motion_features.npy')
            self.pose_mapping = pickle.load(open(f'{args.parsed_dataset_path}/features/raft_motion_features/raft_motion_features.pkl','rb'))
            self.classwise_mapping_pose = defaultdict(list)
            self.classwise_mapping_names_pose = defaultdict(list)
            for k,v in self.pose_mapping.items():
                self.classwise_mapping_pose[k.split('/')[1]].append(v)
                self.classwise_mapping_names_pose[k.split('/')[1]].append(k)
                if len(self.classwise_mapping_names_pose[k.split('/')[0]])>1:
                    last_fr_no = int(self.classwise_mapping_names_pose[k.split('/')[1]][-1].split('/')[-1].split('_')[-1].split('.')[0])
                    second_last_fr_no = int(self.classwise_mapping_names_pose[k.split('/')[1]][-2].split('/')[-1].split('_')[-1].split('.')[0])
                    assert last_fr_no > second_last_fr_no

            for k in self.classwise_mapping_pose.keys():
                self.classwise_mapping_pose[k] = np.stack(self.classwise_mapping_pose[k])

        # Use annotations and create a video set
        self.create_video_set()

        print(f'Num videos {len(self.video_set)}, Source set {self.set}, Mode {self.rmode}')
        self.rng = get_rng(args)

    def get_num_frames(self):
        # query number of total frames in the dataset
        num_frames = np.asarray([a[2] for a in self.video_set])
        return num_frames.mean()

    def read_split_file(self,filename):
        # read split file - based on https://github.com/Sid2697/EgoProceL-egocentric-procedure-learning from ECCV 2022
        with open(filename,'r') as f:
            all_files = f.readlines()
        vid_names = []
        for f in all_files:
            vid_names.append("-".join(f.split(" ")[0].split("-")[:3]))
        
        return list(set(vid_names))

    def gen_labels(self, fps, annotation_data, num_frames, dataset_name=None):
        # get labels - based on https://github.com/Sid2697/EgoProceL-egocentric-procedure-learning from ECCV 2022
        """
        #from egoprocel
        This method is used to generate labels for the test dataset.

        Args:
            fps (int): frame per second of the video
            annotation_data (list): list of procedure steps
            num_frames (int): number of frames in the video

        Returns:
            labels (ndarray): numpy array of labels with length equal to the
                number of frames
        """
        labels = np.zeros(num_frames, dtype=int)
        task_mapping = {}
        for step in annotation_data:
            if dataset_name == 'CrossTask':
                start_time = step[1]
                end_time = step[2]
                label = step[0]
            else:
                start_time = step[0]
                end_time = step[1]
                label = step[2].split()[0]
                task_mapping[label] = ' '.join(step[2].split()[1:])
            start_frame = np.floor(start_time * fps)
            end_frame = np.floor(end_time * fps)
            for count in range(num_frames):
                if count >= start_frame and count <= end_frame:
                    try:
                        labels[count] = int(label)
                    except ValueError:
                        """
                        EGTEA annotations contains key-steps numbers as 1.
                        instead of 1
                        """
                        assert label[-1] == '.'
                        label = label.replace('.', '')
                        labels[count] = int(label)
        return labels, task_mapping

    def create_video_set(self):
        # create a video set using provided annotations. labels are not used for pretraining.
        self.video_set = []
        allowed_video_numbers = list(Path(self.args.parsed_dataset_path).glob("annotations/*.csv"))
        global_task_mapping = {}
        for vid in allowed_video_numbers:
            vid_number = int(str(vid).split('/')[-1].split('.')[0])
            annotation_data  = pd.read_csv(open(str(vid), 'r'),header=None)
            fps = self.fps
            nframes = self.get_num_frames_from_features(vid_number)
            finelabels, task_mapping = self.gen_labels(fps, annotation_data.values, nframes)
            self.video_set.append(
                (
                    str(vid_number),
                    finelabels,
                    nframes
                )
            )
            global_task_mapping.update(task_mapping)
        self.class_names = [(0,'NA')] + [(idx,v) for idx,v in enumerate(global_task_mapping.items())]
        self.num_classes = len(self.class_names)

    def load_pose_from_numpy(self, video_full_path, frame_ind, modality):
        # given frame video name and frame indices, query corresponding features from the pre-extracted features
        if 'raft_motion_features_pose' in modality:
            frame_ind_clipped = np.clip(frame_ind,0,self.classwise_mapping_pose[video_full_path.zfill(4)].shape[0]-1)
            mapped_indices_direct = list(np.take(self.classwise_mapping_pose[video_full_path.zfill(4)],frame_ind_clipped,axis=0))
            pose_from_npy = np.take(self.pose_data, mapped_indices_direct,axis=0)
        
        return pose_from_npy

    def get_clip_vid_new(self,video_id,chunks_to_use=None, random_start=None):
        # sample frames during train/val
        data = self.video_set[video_id]
        n_frames = data[2]
        if self.rmode == 'train':
            num_chunks = self.num_chunks if chunks_to_use is None else chunks_to_use
            extent = int(n_frames)-1
            if random_start is None:
                random_start = self.rng.choice(np.arange(0,n_frames-extent),size=1)[0]
            chunks = np.linspace(random_start,random_start+extent,num_chunks+1)
        else:
            if self.num_chunks_val == -1:
                num_chunks = self.num_chunks
            elif self.num_chunks_val == -2:
                num_chunks = n_frames//self.args.sample_every
            else:
                num_chunks = self.num_chunks_val
            chunks = np.linspace(0,n_frames,num_chunks+1)
        selected_clips = []
        selected_labels = []
        for ch in range(num_chunks):
            if self.rmode == 'train':
                center_frame = self.rng.choice(np.arange(chunks[ch],chunks[ch+1]),1)
            else:
                center_frame = [(chunks[ch] + chunks[ch+1])//2]
            
            selected_f = [int(max(min(f,n_frames-1),0)) for f in center_frame]
            selected_clips.append(selected_f)
            selected_labels.append(data[1][selected_f])
        return (data[0],selected_clips,selected_labels,random_start)

    def __len__(self):
        return len(self.video_set)

    def get_mapped_indices(self,video_full_path,combined_clips,mapping_func=None):
        # given frame video name and frame indices, query corresponding features from the pre-extracted features
        if mapping_func == None:
            mapping_func = self.feature_mapping
        mapped_indices = []
        for cl in combined_clips:
            number = video_full_path.split('/')[-1]
            im_processed = str(cl+1).zfill(5)
            name_to_query = f'{number.zfill(4)}/{im_processed}.jpg'
            mapped_indices.append(mapping_func[name_to_query])
        return mapped_indices
    
    def get_num_frames_from_features(self,key):
        # query number of features
        try:
            feature_mapping_keys = list(self.feature_mapping.keys())
        except:
            feature_mapping_keys = list(self.pose_mapping.keys())
        all_relevant_frames = [k for k in feature_mapping_keys if int(k.split('/')[0]) == key]
        return len(all_relevant_frames)

    def collect_for_getitem(self, video_full_path, frame_ind, modality_to_return='raft_motion_features_pose'):
        window2 = None

        flat_frame_ind_global = [seg[0] for seg in frame_ind]
        imgs, imgs_pose = None, None

        if 'rgb_res50' in modality_to_return:
            mapped_indices_global = self.get_mapped_indices(video_full_path,flat_frame_ind_global)
            imgs = self.feature_data[mapped_indices_global] 
            imgs = torch.from_numpy(imgs).float()

        if 'raft_motion_features_pose' in modality_to_return:
            imgs_pose = self.load_pose_from_numpy(video_full_path, flat_frame_ind_global, modality=modality_to_return)
            imgs_pose = torch.from_numpy(imgs_pose).float()

        return imgs, imgs_pose
        
    def __getitem__(self, index):
        # sample from the dataset
        video_full_path, frame_ind_global1, labels_global1, _ = self.get_clip_vid_new(index)
        frame_ind = [seg[0] for seg in frame_ind_global1]
        imgs, _ = self.collect_for_getitem(video_full_path, frame_ind_global1, modality_to_return='rgb_res50')
        other_modality_features = {}
        for mod in self.args.modalities_to_use.split(',')[1:]:
            _, imgs_pose = self.collect_for_getitem(video_full_path, frame_ind_global1, modality_to_return=mod)
            other_modality_features[mod] = imgs_pose
        
        labels = np.concatenate(labels_global1,0)
        vid_idx = index
        n_frames = self.video_set[index][2]
    
        return_dict = {
            'labels':torch.from_numpy(labels),
            'vid_idx':vid_idx,
            'mid_frame':frame_ind[len(frame_ind)//2],
            'nframes':n_frames,
            'timestamps':torch.from_numpy(np.asarray(frame_ind)),
            'normalized_time1':torch.from_numpy(np.asarray(frame_ind))/n_frames,
            'video_path':video_full_path,
            'fps':self.fps,
            'clip_global':imgs
        }
        return_dict.update(other_modality_features)

        return return_dict

