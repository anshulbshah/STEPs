# based on https://github.com/Sid2697/EgoProceL-egocentric-procedure-learning from ECCV 2022

import torch
import numpy as np
from utils.KSL_utils import run_kmeans, gen_print_results, random_segmentation, uniform_segmentation

class KSL_evaluation:
    def __init__(self,evaluation_approach='kmeans', num_clusters=7, num_keysteps=17, report_perstep=True, skip_background=False):
        # main code to run KSL evaluation. 
        super(KSL_evaluation, self).__init__()
        self.gt = list()
        self.embeddings = list()
        self.average_iou = list()
        self.average_recall = list()
        self.average_precision = list()
        self.ind_preds = list()
        self.average_f1 = list()
        self.num_clusters = num_clusters
        self.num_keysteps = num_keysteps
        assert evaluation_approach in ['kmeans','random','uniform']
        self.evaluation_approach = evaluation_approach
        self.final_report_perstep = report_perstep
        self.skip_background = skip_background
        print(f"Reporting metrics per-step : {self.final_report_perstep}")
        print(f"Num  key steps : {num_keysteps}")
        
    def video_level_metrics(self, embedding, label, video_name):
        # determine video level metrics
        if self.evaluation_approach == 'kmeans':
            ind_preds = run_kmeans(self.num_clusters, embedding)
        elif self.evaluation_approach == 'random':
            ind_preds = random_segmentation(self.num_clusters, embedding)
        elif self.evaluation_approach == 'uniform':
            ind_preds = uniform_segmentation(self.num_clusters, embedding)
        recall, precision, iou, perm_gt, perm_pred = gen_print_results(
                self.num_clusters,
                label.squeeze(),
                ind_preds,
                self.num_keysteps,
                video_name,
                return_assignments=True,
                skip_background=self.skip_background
            )
        self.average_iou.append(iou)
        self.average_recall.append(recall)
        self.average_precision.append(precision)
        f1_score_type3 = 2*recall*precision/(recall+precision+1e-10)
        self.average_f1.append(f1_score_type3)
        self.embeddings.append(embedding)
        self.gt.extend(label)
        self.ind_preds.append(ind_preds)

    def cumulative_metrics(self):
        # get cumulative metrics
        embeddings_ = np.concatenate(self.embeddings, axis=0)
        assert len(self.gt) == embeddings_.shape[0]
        if self.evaluation_approach == 'kmeans':
            overall_preds = run_kmeans(self.num_clusters, embeddings_)
        elif self.evaluation_approach == 'random':
            overall_preds = random_segmentation(self.num_clusters, embeddings_)
        elif self.evaluation_approach == 'uniform':
            overall_preds = uniform_segmentation(self.num_clusters, embeddings_)
        mean_precision, mean_recall, mean_iou, _, mean_fscore_type2 = gen_print_results(
            self.num_clusters,
            torch.from_numpy(np.array(self.gt)),
            overall_preds,
            self.num_keysteps,
            per_keystep=self.final_report_perstep,
            skip_background=self.skip_background
        )
        return mean_precision, mean_recall, mean_iou, _, mean_fscore_type2

if __name__ == '__main__':
    pass
