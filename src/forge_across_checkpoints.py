import models
import math
import random
import argparse as ap
import utils
import tensorflow as tf
from pathlib import Path
import numpy as np
import json
import attacks

def main(args):
    tf.keras.utils.set_random_seed(args.seed)

    if args.deterministic:
        tf.config.experimental.enable_op_determinism()

    run_dir = Path(args.run_dir)
    args.run_dir = str(run_dir.absolute())

    ## load trace's parameters
    with (run_dir / "parameters.json").open('r') as f:
        trace_parameters = json.load(f)

    # setup experiment directory 
    exp_name = Path(__file__).stem.replace('_', '-')
    if not args.name:
        args.name = f"{trace_parameters['name']}__f{args.f}_{args.attack}"
    root, parameters = utils.setup_experiment_directory(args, exp_name)

    input_shape, num_labels, data_loader = utils.DATASETS[trace_parameters["dataset"]]
    global x_train; global y_train
    (x_train, y_train), (x_test, y_test) = data_loader()

    # memoised_path = Path(f"./memoised-datasets/nn-{trace_parameters['dataset']}__n100.npy")
    
    batch_size = trace_parameters["batch_size"]
    ds_len = len(x_train)
    steps_per_epoch = math.ceil(ds_len / batch_size)
    total_steps = trace_parameters["epochs"] * steps_per_epoch
    eta = trace_parameters["eta"]
    checkpoints_dir = run_dir / "checkpoints"


    rng = np.random.default_rng(args.seed)
    true_idx = None
    true_batch_x = None
    true_batch_y = None

    # setup the gradient function
    scce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    @tf.function
    def grad_fn(x, y, model):
        with tf.GradientTape() as tape:
            logits = model(x)
            lossv = model.loss(y, logits)
        g = tape.gradient(lossv, model.trainable_weights)
        return tf.concat([tf.reshape(l, [-1]) for l in g], axis=0)

    # function to measure gradient l2 difference
    @tf.function
    def grad_l2(g1, g2):
        return tf.norm(g1 - g2)


    all_errors = []
    all_losses = []
    all_gnorms = []
    for run in range(args.total_runs):
        true_idx = rng.choice(len(x_train), batch_size)
        true_batch_x = tf.gather(x_train, true_idx)
        true_batch_y = tf.gather(y_train, true_idx)

        print(f"on run {run+1}/{args.total_runs}")
        run_errors = []
        run_losses = []
        run_gnorms = []
        for step in range(0,total_steps+1, args.every):
            model_path = checkpoints_dir / f"weights__{step}.keras"
            model = None
            if model_path.is_file():
                custom_objects=None
                if trace_parameters['model'] == 't200':
                    # save the custom layers
                    custom_objects = {
                        "TransformerBlock" : models.TransformerBlock,
                        "TokenAndPositionEmbedding" : models.TokenAndPositionEmbedding
                    }

                model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
            else:
                raise ValueError(f"No model for step {step} found...")
            
            if model != None:
                # calculate loss
                l = model.loss(true_batch_y[:args.f], model(true_batch_x[:args.f]))
                gnorm = tf.norm(grad_fn(true_batch_x[:args.f], true_batch_y[:args.f], model))
                # calculate the true gradient
                true_grad = grad_fn(true_batch_x, true_batch_y, model)    
                error_fn = lambda x, y: grad_l2(true_grad, grad_fn(x,y, model))
                # if args.attack == 'thudi':
                attack = attacks.ThudiForgingAttack(x_train, y_train, error_fn, seed=args.seed, M=args.M)
                # else:
                #     attack = attacks.ZhangForgingAttack(x_train, y_train, memoised_path)
                
                forged_batch = attack.forge(true_batch_x, true_batch_y, true_idx, args.f)
                e_approx = math.sqrt(eta) * error_fn(*forged_batch)
                print(f"on  step {step}/{total_steps}, {e_approx:.2e}, {l:.2e}, {gnorm:.2e}")
                run_errors.append([step, float(e_approx)])
                run_losses.append([step, float(l)])
                run_gnorms.append([step, float(gnorm)])

        all_errors.append(run_errors)
        all_losses.append(run_losses)
        all_gnorms.append(run_gnorms)

    np.savez(root/ "results.npz",
             approx_errors=np.array(all_errors),
             losses=np.array(all_losses),
             gnorms=np.array(all_gnorms))
            
    parameters["trace_parameters"] = trace_parameters
    with (root / "parameters.json").open('w') as f:
        json.dump(parameters, f, indent=4)


    

    print(f"\n\n{root.absolute()}")



if __name__ == "__main__":
    parser = ap.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--attack", choices=['thudi'], default="thudi")#, 'zhang'])
    parser.add_argument('-f', type=int, default=None, help="Number of examples to forged")
    parser.add_argument('--every', type=int, default=1)
    parser.add_argument('-M', type=int, default=300)
    parser.add_argument('--total-runs', type=int, default=10)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--name", type=str, default="")
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--deterministic", action='store_true', default=False)
    args = parser.parse_args()

    main(args)
