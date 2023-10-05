import torch
import models.models as models
from torch.utils.data import DataLoader
from tqdm import tqdm
import methods.losses as losses
from utils.utils import AverageMeter, ProgressMeter, seed_worker, get_lr
from datasets import Mecanno
from pathlib import Path

def get_lr(optimizer):
    # get current lr
    for param_group in optimizer.param_groups:
        return param_group['lr']

def STEPs_training(args):
    
    # instantiate a custom pytorch dataset
    trainDataset = Mecanno.MeccanoDataset(args=args, set='train', rmode='train')

    sampler = torch.utils.data.RandomSampler(trainDataset, replacement=True, num_samples=max(args.batch_size,len(trainDataset)))

    # create data loader
    trainLoader = DataLoader(trainDataset, batch_size=args.batch_size,num_workers=args.num_workers, drop_last=True,worker_init_fn=seed_worker,sampler=sampler)

    model = models.TemporalEncoder(args,rgb_dim=trainDataset.rgb_dim).cuda()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=0.0)

    if args.lr_drops != '':
        drop_epochs = args.lr_drops.split(',')
        drop_epochs = [int(a) for a in drop_epochs]
        lr_sched = torch.optim.lr_scheduler.MultiStepLR(optimizer, drop_epochs,gamma=0.1)
    else:
        lr_sched = None

    bmc2_loss = losses.bmc2_loss(args.cidm_sigma,2.0,absolute_sigma=True).cuda()
    Path(f'saved_models/iccv2023_STEPs_Meccano').mkdir(exist_ok=True,parents=True)

    for ep in tqdm(range(1,args.num_epoch+1),total=args.num_epoch):
        # iterate for num epochs
        model.train()
        total_loss = AverageMeter('Total Loss', ':.2f')
        temporal_losses = AverageMeter('Temporal Loss v2', ':.2f')
        contrastive_losses = AverageMeter('Contrastive Loss local', ':.2f')

        if lr_sched is not None:
            lr_sched.step()

        progress = ProgressMeter(len(trainLoader),[total_loss],prefix='Epoch:[{}/{}]'.format(ep,args.num_epoch))
        for ii,feat_dict in tqdm(enumerate(trainLoader), total=len(trainLoader),disable=True):
            B = feat_dict['clip_global'].shape[0]
            input_global_rgb = feat_dict['clip_global'].cuda()
            inputs_pose = {}
            for mod in args.modalities_to_use.split(',')[1:]:
                inputs_pose[mod] = feat_dict[mod].cuda()
            
            input_global = (input_global_rgb,inputs_pose)

            # pass sampled raw features through the model
            proj_ftr_seq_rgb, proj_ftr_seq_pose = model(input_global,return_seq_features=True, trainmode=True, timestamps=feat_dict['timestamps'])

            rgb_raw = input_global_rgb[:,:,:-1]
            attention_input = []

            # determine input for the boostrapping loss
            if 'rgb' in args.bcidm_modalities or args.bcidm_modalities == 'all':
                attention_input.append(rgb_raw)
            if 'depth' in args.bcidm_modalities or args.bcidm_modalities == 'all':
                attention_input.append(inputs_pose['depth_res50'])
            if 'gaze' in args.bcidm_modalities or args.bcidm_modalities == 'all':
                attention_input.append(inputs_pose['gaze_pose'])
            if 'raft' in args.bcidm_modalities or args.bcidm_modalities == 'all':
                attention_input.append(inputs_pose['raft_motion_features_pose'])
            attention_input = torch.cat(attention_input,-1)
            normed_input = (attention_input - torch.mean(attention_input,-1,keepdim=True))/(torch.std(attention_input,-1,keepdim=True))
            approach_type = args.temporal_loss_type
            dummy_guide_mask = torch.ones(B,input_global_rgb.shape[1],input_global_rgb.shape[1]-1).to(input_global_rgb)

            # calculate bmc2 loss for rgb-rgb
            temporal_loss_total, all_masks, W_ij_til, W_ij = bmc2_loss(proj_ftr_seq_rgb,sampled_indices=feat_dict['timestamps'],og_seq_len=feat_dict['nframes'],\
                                                                                approach_type=approach_type,
                                                                                per_video_fps=feat_dict['fps'],raw_features=normed_input,guide_mask=dummy_guide_mask) #, pdb=approach_type != 'cidm')

            closs = torch.Tensor([0.0]).to(temporal_loss_total.device)

            #  calculate bmc2 loss for other modalities and cross terms
            for mod in args.modalities_to_use.split(',')[1:]:
                temporal_loss_pose, _,  _, _ =                      bmc2_loss(proj_ftr_seq_pose[mod],precomputed_masks=all_masks,features_other=proj_ftr_seq_pose[mod],suffix='single_m',W_ij_til=W_ij_til, W_ij=W_ij)
                temporal_loss_total = temporal_loss_total + temporal_loss_pose


                temporal_loss_cc_rgb_pose, _, _, _ =                bmc2_loss(proj_ftr_seq_rgb,precomputed_masks=all_masks,features_other=proj_ftr_seq_pose[mod],suffix='cc',W_ij_til=W_ij_til, W_ij=W_ij) #, pdb=approach_type != 'cidm')
                temporal_loss_cc_rgb_pose_other, _, _, _ =          bmc2_loss(proj_ftr_seq_pose[mod],precomputed_masks=all_masks,features_other=proj_ftr_seq_rgb,suffix='cc2',W_ij_til=W_ij_til, W_ij=W_ij) #, pdb=approach_type != 'cidm')

                closs += temporal_loss_cc_rgb_pose + args.lam_mc_anchor*temporal_loss_cc_rgb_pose_other

            loss = args.lam_temporal*temporal_loss_total + args.lam_mc*closs

            temporal_losses.update(temporal_loss_total.item(),B)
            contrastive_losses.update(closs.item(),B)
            total_loss.update(loss.item(),B)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            progress.display(ii)

        metrics_to_log = {
            'losses/total_loss': total_loss.avg,
            'losses/temporal_loss_v2':temporal_losses.avg,
            'losses/contrastive_loss_local':contrastive_losses.avg,
            'learning_rate':get_lr(optimizer)
        }

        if ep%100 == 0:
            torch.save(model.state_dict(),str(Path('saved_models')/"iccv2023_STEPs_Meccano"/(str(ep)+'.pth')))
    return model

if __name__ == '__main__':
    pass