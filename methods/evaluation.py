


import torch

from utils.utils import seed_worker, extract_features_and_add_to_km
import models.models as models
from utils.KSL_metrics import KSL_evaluation
from methods.clustering import *
from datasets import Mecanno
from torch.utils.data import DataLoader
import os
import copy

def cluster_and_evaluate(model,args,valLoader):
    # KSL evaluation.
    model.eval()
    num_clusters = args.num_clusters

    # get all features using the trained temporal encoder and normalize
    km_val = keymoments_kmeans(K=num_clusters,args=args)
    extract_features_and_add_to_km(args,valLoader,model,km_val)
    km_val.finalize_datasets(normalize_features=args.normalize_evaluation)
    
    modalities = args.modalities_to_use.split(',') 
    compute_metrics_for = [(modalities[0],0)]

    stats_val = {}
    for it,(mod_type,idx) in enumerate(compute_metrics_for):
        km_val_to_use = copy.deepcopy(km_val)

        if 'rgb' not in mod_type and (args.test == False and args.eval_all == False):
            continue

        report_perstep = True
        skip_background = False
        # evaluate for KSL
        eval_obj = KSL_evaluation(evaluation_approach='kmeans',num_clusters=args.num_clusters, num_keysteps=valLoader.dataset.num_keysteps, report_perstep=report_perstep, skip_background=skip_background)
        for idx in range(len(km_val_to_use.datasets_separate)):
            eval_obj.video_level_metrics(
                                        embedding = km_val_to_use.datasets_separate[idx].detach().numpy(), \
                                        label = km_val_to_use.ground_truth_labels[idx].detach().numpy(), \
                                        video_name = km_val_to_use.video_names[idx]
                                        )
        mean_precision, mean_recall, mean_iou, _, mean_fscore = eval_obj.cumulative_metrics()
        stats_val[f'{mod_type}_mean_recall'] = mean_recall
        stats_val[f'{mod_type}_mean_precision'] = mean_precision
        stats_val[f'{mod_type}_mean_iou'] = mean_iou
        stats_val[f'{mod_type}_mean_fscore'] = mean_fscore

    print(stats_val)
    return stats_val
    
def evaluate(args, model=None):
    valDataset = Mecanno.MeccanoDataset(args=args, set='test', rmode='val')

    valLoader = DataLoader(valDataset, batch_size=1, shuffle=False,num_workers=0,worker_init_fn=seed_worker)
    
    # either use the model which was just trained, or load a model from a provided checkpoint
    if model is None:
        model = models.TemporalEncoder(args,rgb_dim=valDataset.rgb_dim).cuda()
        if args.load_checkpoint != 'none' and os.path.exists(args.load_checkpoint):
            model.load_state_dict(torch.load(args.load_checkpoint),strict=False)
            print('Loaded model successfully!')

    cluster_and_evaluate(model,args,valLoader)
