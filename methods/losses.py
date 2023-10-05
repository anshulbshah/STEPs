import torch
import torch.nn.functional as F

class bmc2_loss(torch.nn.Module):
    def __init__(self,sigma,margin,absolute_sigma=True):
        super(bmc2_loss, self).__init__()
        self.margin = margin
        self.sigma = sigma
        self.absolute_sigma = absolute_sigma
    
    def calculate_constants(self,seq_len, sampled_indices, device='cpu',per_video_fps=None, raw_ftr_weight=None):
        # obtain the sigma window and boostrap window
        B, S = sampled_indices.shape
        norm_inds = sampled_indices/seq_len.unsqueeze(1)
        W_ij_til_new = (torch.cdist(norm_inds.unsqueeze(-1),norm_inds.unsqueeze(-1))**2 + 1).to(device)
        W_ij_new = 1/W_ij_til_new
        if self.absolute_sigma:
            repeated_sigma_to_use = self.sigma
            absolute_time = sampled_indices/(per_video_fps.unsqueeze(-1).repeat(1,sampled_indices.shape[-1]))
            y_mask_only_temporal_new = ((torch.abs(absolute_time.unsqueeze(-1) - absolute_time.unsqueeze(1)) > repeated_sigma_to_use)).to(device)
        else:
            repeated_sigma_to_use = self.sigma/seq_len.unsqueeze(-1).unsqueeze(-1)
            y_mask_only_temporal_new = ((torch.abs(norm_inds.unsqueeze(-1) - norm_inds.unsqueeze(1)) > repeated_sigma_to_use)).to(device)
        
        if not raw_ftr_weight == None:
            return W_ij_til_new, W_ij_new, raw_ftr_weight, y_mask_only_temporal_new.float()
        else:
            return W_ij_til_new, W_ij_new, None, y_mask_only_temporal_new.float()

    def get_feature_dist_weights(self,feature):
        assert feature.dim() == 3
        feature_dist = torch.cdist(feature.double(),feature.double(),p=2)
        return feature_dist

    def forward(self,features,guide_mask=None,sampled_indices=None, og_seq_len=None, approach_type='cidm', pdb=False,per_video_fps=None,raw_features=None, \
                precomputed_masks=None, features_other=None, suffix='',W_ij_til=None, W_ij=None):
        features = features.contiguous()
        if precomputed_masks is None:
            # the bootstrapped window needs to be computed only once. Other calls to the loss can use precomputed windows
            identity_mask = (torch.eye(guide_mask.shape[-2]).unsqueeze(0).repeat(guide_mask.shape[0],1,1).bool()).to(guide_mask.device)
            raw_features = raw_features.contiguous()
            raw_ftr_weight = self.get_feature_dist_weights(raw_features)
            W_ij_til, W_ij, self_distances, y_mask_only_temporal = self.calculate_constants(og_seq_len,sampled_indices,device=features.device,per_video_fps=per_video_fps,raw_ftr_weight=raw_ftr_weight)
            self_distances_nodiag = torch.masked_select(self_distances,~identity_mask).view(guide_mask.shape)
                
            y_mask_only_temporal_without_self = torch.masked_select(y_mask_only_temporal,~identity_mask).view(guide_mask.shape)
            one_minus_y_mask_only_temporal_without_self = 1.0 - y_mask_only_temporal_without_self

            threshold = torch.sum(one_minus_y_mask_only_temporal_without_self*self_distances_nodiag,-1,keepdim=True)/(torch.sum(one_minus_y_mask_only_temporal_without_self,-1,keepdim=True) + 1E-8).repeat(1,1,one_minus_y_mask_only_temporal_without_self.shape[-1])
            
            guide_mask_thresholded = self_distances_nodiag <= threshold
            guide_mask_thresholded = (~guide_mask_thresholded).float()            

            if approach_type == 'cidm':
                # no bootstrapping
                mask_to_use_positive = y_mask_only_temporal
                mask_to_use_negative = 1.0 - mask_to_use_positive
            
            elif approach_type == 'cidm_OR_new_window':
                # with bootstrapping
                new_mask = identity_mask.float()
                new_mask[~identity_mask] = guide_mask_thresholded.view(-1)         
                mask_to_use_positive = (~torch.logical_or(~y_mask_only_temporal.bool(),~new_mask.bool())).float()
                mask_to_use_negative = 1.0 - mask_to_use_positive

            inverted_mask_positive = 1.0 - mask_to_use_positive
            inverted_mask_negative = 1.0 - mask_to_use_negative
            feature_dist = torch.cdist(features,features,p=2)

        else:
            inverted_mask_positive = precomputed_masks[0]
            inverted_mask_negative = precomputed_masks[1]
            y_mask_only_temporal = precomputed_masks[2]
            feature_dist = torch.cdist(features,features_other,p=2)
            mask_to_use_positive = None

        loss = inverted_mask_negative*W_ij_til*torch.relu(self.margin - feature_dist) + inverted_mask_positive*W_ij*feature_dist
        loss = loss.mean()

        return loss, (inverted_mask_positive, inverted_mask_negative, y_mask_only_temporal), W_ij_til, W_ij