import json
from pathlib import Path
import sys
import itertools
import models
import tensorflow as tf
import dataset_utils as du
from functools import reduce
from operator import mul 
import platform
import datetime as dt
import subprocess


MODELS = {
    "lenet" : models.LeNet,
    "vgg" : models.VGGmini,
    "fcn1": models.FCN_single,
    "t200" : models.Transformer # 200 context-window
}

DATASETS = {
    "mnist"   : ((28,28,1), 10, du.load_and_process_mnist),
    "cifar10" : ((32,32,3), 10, du.load_and_process_cifar10),
    "adult" : ((14,), 1, du.load_and_process_adult),
    "imdb" : ((200,), 1, du.load_and_process_imdb)
}



def setup_experiment_directory(args, exp_name):
    # set experiment timestamp
    TIMESTAMP = dt.datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    # create experiments results folder


    parameters = vars(args)
    parameters["command"] = ' '.join(sys.argv)
    parameters["cpu"] = platform.processor()
    parameters["devices"] = [d.name for d in tf.config.list_physical_devices()]
    parameters["cuda"] = tf.test.is_built_with_cuda()
    parameters["machine"] = platform.node()
    if tf.test.is_built_with_cuda():
        gpus = subprocess.check_output("nvidia-smi -L", shell=True).decode("ascii").split('\n')
        parameters["gpus"] = gpus
    else:
        parameters["gpus"] = []
    
    print(parameters)

    name_suffix = '-' + args.name if args.name and args.name[0] != '_' else ''
    root = Path(f"experiments/{TIMESTAMP}--{exp_name}{name_suffix}")
    root.mkdir(parents=True)
    with (root / "parameters.json").open('w') as f:
        json.dump(parameters, f, indent=4)
    
    return root, parameters


def update_params(rd, new_params):
    with (rd / "parameters.json").open('w') as f:
        json.dump(new_params, f, indent=4)


def load_params(rd):
    with (rd / "parameters.json").open('r') as f:
        p = json.load(f)
    return p
        
