import numpy as np
import torch

class keymoments_kmeans:
    def __init__(self,K,args):
        self.K = K
        self.datasets = []
        self.timestamps = []
        self.dataset_number = []
        self.dataset_lengths = []
        self.ground_truth_labels = []
        self.video_names = []
        self.dataset_counter = 0
        self.flat_labels = []
        self.subsampled_segment_list = []
        self.args = args

    def add_dataset(self,data,timestamps,gtlabels,names,subsampled_segment_list):
        # add a video to the KM object
        self.datasets.append(data[0])
        self.timestamps.append(timestamps[0])
        self.dataset_number.append(self.dataset_counter*torch.ones(data[0].shape[0]))
        self.dataset_lengths.append(data[0].shape[0])
        self.video_names.append(names)
        self.subsampled_segment_list.append(subsampled_segment_list)
        if gtlabels is not None:
            self.ground_truth_labels.append(gtlabels[0])
        self.dataset_counter += 1

    def normalize_datasets(self):
        # normalize features for each video
        self.datasets = torch.nn.functional.normalize(self.datasets,p=2,dim=-1)
        for el in range(len(self.datasets_separate)):
            self.datasets_separate[el] = torch.nn.functional.normalize(self.datasets_separate[el],p=2,dim=-1)
    
    def finalize_datasets(self,normalize_features='False'):
        # combine into a big dataset
        self.datasets_separate = self.datasets.copy()
        self.datasets = torch.cat(self.datasets)
        self.dataset_number = torch.cat(self.dataset_number)
        for el in self.ground_truth_labels:
            self.flat_labels.extend(list(el))
        self.flat_labels = np.asarray(self.flat_labels)
        if normalize_features == 'True':
            print('Normalizing features')
            self.normalize_datasets()

    def __repr__(self):
        print(f'Num videos : {len(self.datasets_separate)}')
        print(f'Datasets : {self.datasets.shape}')
        datasets_separate_lengths = [a.shape[0] for a in self.datasets_separate]
        assert sum(datasets_separate_lengths) == sum(self.dataset_lengths)
        flat_labels = []
        for a in self.ground_truth_labels:
            flat_labels.extend(list(a.numpy()))
        print(f'Datasets separate : {sum(datasets_separate_lengths)}')
        print(f'Datasets lengths : {sum(datasets_separate_lengths)}')
        print(f'Label lengths {len(flat_labels)}')
        print(f'Labels min {min(flat_labels)} max {max(flat_labels)} labels all {set(flat_labels)}')
        return "" 
