import methods.selfsup as selfsup
import methods.evaluation as evaluation
import opts
from utils.utils import set_random_seed
if __name__ == '__main__':
    args = opts.parse_args() 
    set_random_seed(args.random_seed)

    assert args.train or args.test, "Set atleast one of train or test"
    trained_model = None
    if args.train:
        trained_model = selfsup.STEPs_training(args)
    if args.test:   
        evaluation.evaluate(args,model=trained_model)
   
    