import argparse
def parse_args():    
    parser = argparse.ArgumentParser(description='Runner script')

    # Dataset    
    parser.add_argument("--parsed_dataset_path",type=str,default='Data/Meccano', help="Parsed features location")

    # Training
    parser.add_argument("--modalities_to_use",type=str,default="rgb_res50,raft_motion_features_pose", help="Comma separated modalities to train with")
    parser.add_argument("--num_chunks",type=int,default=1024, help="Num Chunks to sample")
    parser.add_argument("--batch_size",type=int,default=4,help="Batch Size")
    parser.add_argument("--num_epoch",type=int,default=300,help="Num epochs")
    parser.add_argument("--learning_rate",type=float,default=1E-3,help="LR")
    parser.add_argument("--random_seed",type=int,default=0,help="Set a random seed")
    parser.add_argument("--num_workers",type=int,default=4,help="Num workers to train with")
    parser.add_argument("--lr_drops",type=str,default='',help="Comma separated epochs to drop lr by 0.1x")

    # Model
    parser.add_argument("--nlayers",type=int,default=2,help="Num layers")
    parser.add_argument("--nheads",type=int,default=2,help="Number of transformer heads")
    parser.add_argument("--nhid",type=int,default=128,help="Size of hidden layers in temporal model")
    parser.add_argument("--train",action='store_true',help="Train model")

    # loss
    parser.add_argument("--lam_temporal",type=float,default=1.0, help="lambda_(u-u)")
    parser.add_argument("--lam_mc",type=float,default=1.0, help="lambda_(rgb-v)")
    parser.add_argument("--lam_mc_anchor",type=float,default=1.0, help="lambda_(v-rgb)")
    parser.add_argument("--cidm_sigma",type=float,default=10,help="size of the sigma window")
    parser.add_argument("--temporal_loss_type",type=str,default='cidm_OR_new_window',help="Window to use")
    parser.add_argument("--bcidm_modalities",type=str,default="rgb", help="Modality to use for bootstrapping")

    # evaluation
    parser.add_argument("--test",action='store_true',help="Test model")
    parser.add_argument("--load_checkpoint",type=str,default="none",help="Location to trained checkpoint")
    parser.add_argument("--normalize_evaluation",type=str,default='True', help="How to normalize the input features")
    parser.add_argument("--num_chunks_val",type=int,default=-2, help="Num chunks to sample for val")
    parser.add_argument("--sample_every",type=int,required=False,default=2,help="Frame sampling rate for evaluation")
    parser.add_argument("--num_clusters",type=int,default=7, help="Number of Clusters")


    args = parser.parse_args()

    return args

if __name__ == '__main__':
    pass
