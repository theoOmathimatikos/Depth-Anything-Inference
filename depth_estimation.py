import argparse
from Depth_Anything.metric_depth.evaluate import custom_infer
from Depth_Anything.metric_depth.zoedepth.utils.arg_utils import parse_unknown

# Should do the same with Depth_Anything_2


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str,
        default="zoedepth", help="Name of the model to evaluate")
    parser.add_argument("-p", "--pretrained_resource", type=str,
        default="local::./Depth_Anything/metric_depth/checkpoints/depth_anything_metric_depth_outdoor.pt")
    parser.add_argument("-d", "--dataset", type=str, required=False,
        default='custom_outdoor')
    
    args, unknown_args = parser.parse_known_args()
    overwrite_kwargs = parse_unknown(unknown_args)

    print(args.pretrained_resource)

    custom_infer(args.pretrained_resource, args.model, 
                 args.dataset, **overwrite_kwargs)