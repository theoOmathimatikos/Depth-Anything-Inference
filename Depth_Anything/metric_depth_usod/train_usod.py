# MIT License

# Copyright (c) 2022 Intelligent Systems Lab Org

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# File author: Shariq Farooq Bhat

from zoedepth.utils.misc import count_parameters, parallelize
from zoedepth.utils.config import get_config
from zoedepth.utils.arg_utils import parse_unknown
from zoedepth.trainers.builder import get_trainer
from zoedepth.models.builder import build_model
from zoedepth.data.data_mono import DepthDataLoader
import torch.utils.data.distributed
import torch.multiprocessing as mp
import torch
import numpy as np
import glob
import os
import random
from zoedepth.models.model_io import load_wts

os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ["WANDB_START_METHOD"] = "thread"


def fix_random_seed(seed: int):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def load_ckpt(config, model, checkpoint_dir="./checkpoints", ckpt_type="best"):

    if hasattr(config, "checkpoint"):
        checkpoint = config.checkpoint  + "/kaggle/input/depth_anything/torch/version/1/models/"

    elif hasattr(config, "ckpt_pattern"):
        
        pattern = config.ckpt_pattern
        matches = glob.glob(os.path.join(
            checkpoint_dir, f"*{pattern}*{ckpt_type}*"))
        if not (len(matches) > 0):
            raise ValueError(f"No matches found for the pattern {pattern}")

        checkpoint = matches[0]

    else:
        return model
    
    model = load_wts(model, checkpoint)
    print("Loaded weights from {0}".format(checkpoint))
    return model


def main_worker(config):

    # try:

    seed = config.seed if 'seed' in config and config.seed else 43
    fix_random_seed(seed)

    # print("Before model loading:", torch.cuda.memory_allocated() / 1024**2, 'MB') 
    model = build_model(config)
    model = load_ckpt(config, model)
    model = parallelize(config, model)

    # print("After model loading:", torch.cuda.memory_allocated() / 1024**2, 'MB') 

    total_params = f"{round(count_parameters(model)/1e6,2)}M"
    config.total_params = total_params
    print(f"Total parameters : {total_params}")

    train_loader = DepthDataLoader(config, "train").data
    test_loader = DepthDataLoader(config, "online_eval").data

    trainer = get_trainer(config)(
        config, model, train_loader, test_loader, device=config.gpu)

    trainer.train()
        
    # finally:
    #     import wandb
    #     wandb.finish()


def standardize_config(config):

    config.shared_dict = None

    config.batch_size = 1
    config.workers = 3
    config.mode = 'train'
    config.world_size = 1
    config.rank = 0

    if config.root != "." and not os.path.isdir(config.root):
        os.makedirs(config.root)

    ngpus_per_node = torch.cuda.device_count()
    config.num_workers = config.workers
    config.ngpus_per_node = ngpus_per_node
    config.distributed = False
    config.multigpu = False
    config.gpu = 0
    config.save_dir = "kaggle/working"
    
    return config



def call_train_process(num_epochs=10):
    
    # mp.set_start_method('forkserver')
    # mp.set_start_method('fork')

    args = {"dataset": "usod10k", "trainer": None, "model": "zoedepth"}
    overwrite_kwargs = {"pretrained_resource": "/kaggle/input/depth_anything/pytorch/default/1/checkpoints/depth_anything_metric_depth_outdoor.pt", 
                      "model": "zoedepth"}

    config = get_config(args['model'], "train", args['dataset'], **overwrite_kwargs)
    config.epochs = num_epochs
    config = standardize_config(config)

    # if config.use_shared_dict:
    #     shared_dict = mp.Manager().dict()

    # SLURM thrown out

    # nodes = ["127.0.0.1"]
    # if config.distributed:

    #     print(config.rank)
    #     port = np.random.randint(15000, 15025)
    #     config.dist_url = 'tcp://{}:{}'.format(nodes[0], port)
    #     print(config.dist_url)
    #     config.dist_backend = 'nccl'
    #     config.gpu = None



    # if config.distributed:
    #     config.world_size = ngpus_per_node * config.world_size
    #     mp.spawn(main_worker, nprocs=ngpus_per_node,
    #              args=(ngpus_per_node, config))
    # else:

    main_worker(config)
