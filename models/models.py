import torch
import models.transformer as transformer
class TemporalEncoder(torch.nn.Module):
    # Temporal Encoder + MLP
    def __init__(self,args=None, rgb_dim=2048, subtype='default'):
        super(TemporalEncoder, self).__init__()

        # instantiate a transformer for RGB
        self.rgb_temporal_encoder = temporal_model_chooser(args,ninp=rgb_dim, nhead=args.nheads, nhid=args.nhid, nlayers=args.nlayers, dropout=0.0)
        seq_out_in_ftrs = args.nhid
        self.rgb_mlp = torch.nn.Sequential(
            torch.nn.Linear(seq_out_in_ftrs,128),
            torch.nn.ReLU(),
            torch.nn.Linear(128,128)
        )

        self.pose_other_modalities = args.modalities_to_use.split(',')[1:]
        seq_out_in_ftrs_pose = args.nhid

        # instantiate a transformers for other modalities
        if 'raft' in args.modalities_to_use:
            raft_d_pose = 128
            self.raft_temporal_encoder = temporal_model_chooser(args,ninp=raft_d_pose, nhead=args.nheads, nhid=args.nhid, nlayers=args.nlayers, dropout=0.0)
            self.raft_pose_mlp = torch.nn.Sequential(
                torch.nn.Linear(seq_out_in_ftrs_pose,128),
                torch.nn.ReLU(),
                torch.nn.Linear(128,128)
            )

        if 'gaze' in args.modalities_to_use :
            gaze_d_pose = 48
            self.gaze_temporal_encoder = temporal_model_chooser(args,ninp=gaze_d_pose, nhead=args.nheads, nhid=args.nhid, nlayers=args.nlayers, dropout=0.0)
            self.gaze_pose_mlp = torch.nn.Sequential(
                torch.nn.Linear(seq_out_in_ftrs_pose,128),
                torch.nn.ReLU(),
                torch.nn.Linear(128,128)
            )

        if 'depth' in args.modalities_to_use:
            depth_d_pose = rgb_dim
            self.depth_temporal_encoder = temporal_model_chooser(args,ninp=depth_d_pose, nhead=args.nheads, nhid=args.nhid, nlayers=args.nlayers, dropout=0.0)
            self.depth_pose_mlp = torch.nn.Sequential(
                torch.nn.Linear(seq_out_in_ftrs_pose,128),
                torch.nn.ReLU(),
                torch.nn.Linear(128,128)
            )

        self.subtype = subtype
        self.args = args
    def forward(self, inp_features, return_seq_features, trainmode=False, timestamps=None):
        # forward through the model and get adapted features
        assert return_seq_features == True
        rgb_input, pose_inputs = inp_features

        out = self.rgb_temporal_encoder(rgb_input[:,:,:-1].permute(1,0,2),None,timestamps=timestamps)
        rgb_seq_out = out[1][:-1].permute(1,0,2)
        rgb_seq_ftr = self.rgb_mlp(rgb_seq_out)
        rgb_seq_proj = torch.nn.functional.normalize(rgb_seq_ftr, dim=-1, p=2)

        self.pose_seq_out = {}
        self.pose_seq_proj = {}

        if 'depth_res50' in self.pose_other_modalities:
            out = self.depth_temporal_encoder(pose_inputs['depth_res50'][:,:,:-1].permute(1,0,2),None,timestamps=timestamps)
            self.pose_seq_out['depth_res50'] = out[1][:-1].permute(1,0,2)
            self.pose_seq_proj['depth_res50'] = torch.nn.functional.normalize(self.depth_pose_mlp(self.pose_seq_out['depth_res50']), dim=-1, p=2)
        
        if 'raft_motion_features_pose' in self.pose_other_modalities:
            out = self.raft_temporal_encoder(pose_inputs['raft_motion_features_pose'].permute(1,0,2),None,timestamps=timestamps)
            self.pose_seq_out['raft_motion_features_pose'] = out[1][:-1].permute(1,0,2)
            self.pose_seq_proj['raft_motion_features_pose'] = torch.nn.functional.normalize(self.raft_pose_mlp(self.pose_seq_out['raft_motion_features_pose']), dim=-1, p=2)

        if 'gaze_pose' in self.pose_other_modalities:
            out = self.gaze_temporal_encoder(pose_inputs['gaze_pose'].permute(1,0,2),None,timestamps=timestamps)
            self.pose_seq_out['gaze_pose'] = out[1][:-1].permute(1,0,2)
            self.pose_seq_proj['gaze_pose'] = torch.nn.functional.normalize(self.gaze_pose_mlp(self.pose_seq_out['gaze_pose']), dim=-1, p=2)

        if trainmode:
            return rgb_seq_proj, self.pose_seq_proj
        else:
            return None, \
                None, \
                rgb_seq_out, \
                rgb_seq_proj

def temporal_model_chooser(args,ninp,nhead,nhid,nlayers,dropout=0.0,bidirectional=True,channels=None):
    return transformer.TransformerModel(ninp,nhead,nhid,nlayers,dropout,args)

if __name__ == '__main__':
    pass