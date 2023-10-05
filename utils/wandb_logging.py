import wandb
from datetime import datetime

def make_table(columns,rows):
    return wandb.Table(columns=columns, data=rows)

def initialize(args):
    if args.wandb_logging == 'True':
        if args.wandb_tags == '':
            tags = None
        else:
            tags = args.wandb_tags.split(',')
        wandb.init(anonymous="allow",config=args,tags=tags,settings=wandb.Settings(start_method='thread'))
        args.wandb_name = wandb.run.name
    else:
        now = datetime.now()
        args.wandb_name = now.strftime("%d_%m_%Y_%H_%M_%S")

def log_data(args,data_dict,step):
    if args.wandb_logging == 'True':
        wandb.log(data_dict,step=step)

