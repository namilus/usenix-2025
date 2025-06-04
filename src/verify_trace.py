import numpy as np
import math
from pathlib import Path
import argparse as ap
import utils
import tensorflow as tf
import json
import models


def l2_error(weights1, weights2):
    weights1 = tf.concat([tf.reshape(w, [-1]) for w in weights1], axis=0)
    weights2 = tf.concat([tf.reshape(w, [-1]) for w in weights2], axis=0)
    return tf.norm(weights1 - weights2)

def l_inf_error(weights1, weights2):
    weights1 = tf.concat([tf.reshape(w, [-1]) for w in weights1], axis=0)
    weights2 = tf.concat([tf.reshape(w, [-1]) for w in weights2], axis=0)
    return tf.norm(weights1 - weights2, ord=np.inf)

class VerifyCallback(tf.keras.callbacks.Callback):
    def __init__(self, idx, checkpoints_dir, every, mods, custom_objects=None):
        super().__init__()
        self.idx = idx
        self.checkpoints_dir = checkpoints_dir
        self.every = every
        self.mods = set([0] + [int(self.every/m) if m !=0 else 0 for m in mods])
        self.current_step = 0
        self.l2_recomp_errors = []
        self.l_inf_recomp_errors = []
        self.custom_objects = custom_objects

    def calc_model_error(self):
        curr_model_path = self.checkpoints_dir / f"weights__{self.current_step}.keras"
        curr_model = tf.keras.models.load_model(curr_model_path, custom_objects=self.custom_objects)
        l2_dist = l2_error(curr_model.trainable_weights, self.model.trainable_weights)
        l_inf_dist = l_inf_error(curr_model.trainable_weights, self.model.trainable_weights)
        self.l2_recomp_errors.append([self.current_step, l2_dist])
        self.l_inf_recomp_errors.append([self.current_step, l_inf_dist])

        print(f"step {self.current_step} : {l2_dist:.2e}")

    def load_model(self, step):
        step_model_path = self.checkpoints_dir / f"weights__{step}.keras"
        #print("Loaded model", step_model_path)
        self.model.load_weights(step_model_path)

    def on_train_batch_begin(self, batch, logs=None):
        # increment the current step counter
        self.current_step +=1

        if self.every == 0:
            return

        # if this is the step after one that we checked, we should
        # load the previous step's model
        if self.current_step > 1:
            mod = (self.current_step - 1) % self.every
            # only reload if is the k-th checkpoint and not a mod one
            # used for understanding how fast the accumulation occurs
            if mod in self.mods and mod == 0: 
                # load this model before training because we need to
                # reload each model
                self.load_model(self.current_step - 1)
            
            
    def on_train_batch_end(self, batch, logs=None):
        if self.every == 0:
            return
        
        mod = self.current_step % self.every
        if mod in self.mods:
            self.calc_model_error()

    def on_train_begin(self, logs=None):
        # load the first model
        self.load_model(self.current_step)

    def on_train_end(self, logs=None):
        # verify the last model
        self.calc_model_error()


def main(args):
    if args.deterministic:
        tf.config.experimental.enable_op_determinism()

    run_dir = Path(args.run_dir)
    args.run_dir = str(run_dir.absolute())

    ## load trace's parameters
    with (run_dir / "parameters.json").open('r') as f:
        trace_parameters = json.load(f)



    if not args.name:
        args.name = trace_parameters["name"] + f"_tr{args.total_runs}"
    # setup experiment directory
    exp_name = Path(__file__).stem.replace('_', '-')
    root, parameters = utils.setup_experiment_directory(args, exp_name)


    # set the random seed
    tf.keras.utils.set_random_seed(trace_parameters["seed"])

    # load the PoL's dataset
    input_shape, num_labels, data_loader = utils.DATASETS[trace_parameters["dataset"]]
    (x_train, y_train), (x_test, y_test) = data_loader()

    all_recomp_errors_l2 = []
    all_recomp_errors_l_inf = []

    run_histories = []

    for run in range(args.total_runs):
        print(f"\n\t*Recomputation ({run+1}/{args.total_runs})")
        model, loss_fn = utils.MODELS[trace_parameters["model"]](input_shape, num_labels)


        if trace_parameters["model"] == "t200":
            optimiser = tf.keras.optimizers.Adam(trace_parameters["eta"])
        else:
            optimiser = tf.keras.optimizers.SGD(trace_parameters["eta"])

        model.compile(loss=loss_fn, optimizer=optimiser,
                      metrics=["accuracy"])

        custom_objects=None
        if trace_parameters['model'] == 't200':
            # save the custom layers
            custom_objects = {
                "TransformerBlock" : models.TransformerBlock,
                "TokenAndPositionEmbedding" : models.TokenAndPositionEmbedding
            }

        verify_cb = VerifyCallback(run, run_dir / "checkpoints",
                                   trace_parameters["every"],
                                   trace_parameters["mods"],
                                   custom_objects=custom_objects)


        history = model.fit(x_train,
                            y_train,
                            batch_size=trace_parameters["batch_size"],
                            epochs=trace_parameters["epochs"],
                            validation_data=(x_test, y_test),
                            callbacks=[verify_cb])

        run_histories.append(history.history)


        all_recomp_errors_l2.append(verify_cb.l2_recomp_errors)
        all_recomp_errors_l_inf.append(verify_cb.l_inf_recomp_errors)        

        
    print(np.array(all_recomp_errors_l2).shape)
    print(np.array(all_recomp_errors_l2))
    

    np.savez(root / "recomp_errors_l2.npz", recomp_errors=np.array(all_recomp_errors_l2))
    np.savez(root / "recomp_errors_l_inf.npz", recomp_errors=np.array(all_recomp_errors_l_inf))

    # save run histories
    with(root / "run_histories.json").open('w') as f:
        json.dump(run_histories, f, indent=4)


    ## save some the generator machine parameters
    parameters["generator_params"] = trace_parameters

    with (root / "parameters.json").open('w') as f:
        json.dump(parameters, f, indent=4)

    print(f"\n\n{root.absolute()}")


if __name__ == "__main__":
    parser = ap.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--name", type=str, default="")
    parser.add_argument("--total-runs", type=int, default=1)
    parser.add_argument("--deterministic", action='store_true', default=False)
    args = parser.parse_args()
    main(args)
