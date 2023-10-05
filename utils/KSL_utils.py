
# based on https://github.com/Sid2697/EgoProceL-egocentric-procedure-learning
import torch
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from scipy.optimize import linear_sum_assignment
def compute_align_MoF_UoI(
    keystep_pred,
    keystep_gt,
    n_keystep,
    M=None,
    per_keystep=False,
    return_assignments=False,
    skip_background=False
):
    # compute KSL metrics
    try:
        keystep_pred = torch.FloatTensor(keystep_pred)
    except:
        pass
    if skip_background:
        if type(keystep_pred) == torch.Tensor:
            keystep_pred = keystep_pred.detach().cpu().numpy()
        Z_pred = torch.eye(M)[keystep_pred.astype(np.int32), :].float().cpu().numpy()
        Z_gt = torch.eye(n_keystep)[keystep_gt, :].float().cpu().numpy()
        Z_gt = Z_gt[:,1:]
    else:
        if type(keystep_pred) == torch.Tensor:
            keystep_pred = keystep_pred.detach().cpu().numpy()
        Z_pred = torch.eye(M)[keystep_pred.astype(np.int32), :].float().cpu().numpy()
        Z_gt = torch.eye(n_keystep)[keystep_gt, :].float().cpu().numpy()

    assert Z_pred.shape[0] == Z_gt.shape[0]
    T = Z_gt.shape[0]*1.0

    Dis = 1.0 - np.matmul(np.transpose(Z_gt), Z_pred)/T

    perm_gt, perm_pred = linear_sum_assignment(Dis)
    Z_pred_perm = Z_pred[:, perm_pred]
    Z_gt_perm = Z_gt[:, perm_gt]

    if per_keystep:
        list_MoF = []
        list_IoU = []
        list_precision = []
        step_wise_metrics = dict()
        for count, idx_k in enumerate(range(Z_gt_perm.shape[1])):
            pred_k = Z_pred_perm[:, idx_k]
            gt_k = Z_gt_perm[:, idx_k]

            intersect = np.multiply(pred_k, gt_k)
            union = np.clip((pred_k + gt_k).astype(float), 0, 1)

            n_intersect = np.sum(intersect)
            n_union = np.sum(union)
            n_predict = np.sum(pred_k)

            n_gt = np.sum(gt_k == 1)

            if n_gt != 0:
                MoF_k = n_intersect/n_gt
                IoU_k = n_intersect/n_union
                if n_predict == 0:
                    Prec_k = 0
                else:
                    Prec_k = n_intersect/n_predict
            else:
                MoF_k, IoU_k, Prec_k = [-1, -1, -1]
            list_MoF.append(MoF_k)
            list_IoU.append(IoU_k)
            list_precision.append(Prec_k)
            step_wise_metrics[count] = {
                "MoF": MoF_k,
                "IoU": IoU_k,
                "prec": Prec_k
            }

        arr_MoF = np.array(list_MoF)
        arr_IoU = np.array(list_IoU)
        arr_prec = np.array(list_precision)

        mask = arr_MoF != -1
        MoF = np.mean(arr_MoF[mask])
        IoU = np.mean(arr_IoU[mask])
        Precision = np.mean(arr_prec[mask])
        fscores_type2 = 2*MoF*Precision/(MoF+Precision+1e-10)
        if return_assignments:
            return None, None, None, perm_gt, perm_pred
        else:
            return MoF, IoU, Precision, step_wise_metrics, None, fscores_type2
    else:
        intersect = np.multiply(Z_pred_perm, Z_gt_perm)
        union = np.clip((Z_pred_perm + Z_gt_perm).astype(float), 0, 1)

        n_intersect = np.sum(intersect)
        n_union = np.sum(union)
        n_predict = np.sum(Z_pred_perm)

        n_gt = np.sum(Z_gt_perm)

        MoF = n_intersect/n_gt
        IoU = n_intersect/n_union
        Precision = n_intersect/n_predict
        f1_score =  2*MoF*Precision/(MoF+Precision+1e-10)
    if return_assignments:
        return None, None, None, perm_gt, perm_pred
    else:
        return MoF, IoU, Precision, None, f1_score, f1_score


def run_kmeans(num_clusters, features):
    # run kmeans
    kmeans = MiniBatchKMeans(
        n_clusters=num_clusters,
        init='k-means++',
        max_no_improvement=None
    ).fit(features)
    kmeans_preds = kmeans.labels_.copy()
    return kmeans_preds

def random_segmentation(num_clusters, features):
    # run random segmentation
    L = num_clusters
    random_predictions = np.random.randint(L, size=(features.shape[0],))
    return random_predictions

def uniform_segmentation(num_clusters, features):
    # run uniform segmentation
    L = num_clusters
    uniform_predictions = np.linspace(0,L-1,features.shape[0]).astype(int)
    return uniform_predictions

def gen_print_results(
    num_clusters,
    gt,
    pred,
    num_keysteps,
    video_name=None,
    per_keystep=False,
    return_assignments=False,
    skip_background=False
):
    recall, IoU, precision, step_wise_metrics, fscore, fscore_type2 = compute_align_MoF_UoI(
        pred,
        gt,
        num_keysteps + 1,
        M=num_clusters,
        per_keystep=per_keystep,
        skip_background=skip_background
    )
    if video_name:
        if return_assignments:
            _, _, _, perm_gt, perm_pred = compute_align_MoF_UoI(
                pred,
                gt,
                num_keysteps + 1, 
                M = num_clusters,
                per_keystep=per_keystep,
                return_assignments=return_assignments,
                skip_background=skip_background
            )
            return recall, precision, IoU, perm_gt, perm_pred
        return recall, precision, IoU
    else:
        print(
            f"Overall Results. Precision: {precision}, Recall: {recall}, IOU: "
            f"{IoU}")
        return precision, recall, IoU, fscore, fscore_type2
