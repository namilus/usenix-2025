import json
from pathlib import Path
import argparse as ap
import utils
import tensorflow as tf
import math
class GenerateTraceCallback(tf.keras.callbacks.Callback):
    def __init__(self, checkpoints_dir, every, mods):
        super().__init__()
        self.checkpoints_dir = checkpoints_dir
        self.every = every
        self.mods = set([0] +[m if m !=0 else 0 for m in mods])
        self.current_step = 0

    def save_model(self):
        self.model.save(self.checkpoints_dir / f"weights__{self.current_step}.keras")

    def on_train_batch_end(self, batch, logs=None):
        self.current_step += 1
        if self.every == 0:
            return
        
        mod = self.current_step % self.every
        if mod in self.mods:
            self.save_model()

    def on_train_begin(self, logs=None):
        self.save_model()

    def on_train_end(self, logs=None):
        self.save_model()
        

def main(args):
    # set the random seed
    tf.keras.utils.set_random_seed(args.seed)
    if args.deterministic:
        tf.config.experimental.enable_op_determinism()

    # setup experiment directory
    exp_name = Path(__file__).stem.replace('_', '-')
    if args.name == "":
        args.name = f"{args.model}-{args.dataset}"
        args.name += f"__b{args.batch_size}_e{args.epochs}_k{args.every}_seed{args.seed}"
    root, parameters = utils.setup_experiment_directory(args, exp_name)
    checkpoints_dir = (root / "checkpoints")
    checkpoints_dir.mkdir(parents=True)


    input_shape, num_labels, data_loader = utils.DATASETS[args.dataset]
    (x_train, y_train), (x_test, y_test) = data_loader()

    model, loss_fn = utils.MODELS[args.model](input_shape, num_labels)

    # write model summary to file and configuration
    with (root / "model_summary.txt").open('w') as f:
        model.summary(print_fn=lambda x: print(x, file=f))

    with (root /  "model_architecture.json").open('w') as f:
        print(model.to_json(indent=4), file=f)

    if args.model == 't200':
        optimiser = tf.keras.optimizers.Adam(args.eta)
    else:
        optimiser = tf.keras.optimizers.SGD(args.eta)

    model.compile(loss=loss_fn, optimizer=optimiser,
                  metrics=["accuracy"])


    trace_cb = GenerateTraceCallback(checkpoints_dir, args.every, args.mods)


    history = model.fit(x_train, y_train,
                        batch_size=args.batch_size,
                        epochs=args.epochs,
                        validation_data=(x_test, y_test),
                        callbacks=[trace_cb])


    model.save(checkpoints_dir / "final.keras")
    with (root / "history.json").open('w') as f:
        json.dump(history.history, f, indent=4)

    parameters["total_steps"] = trace_cb.current_step
    utils.update_params(root, parameters)

    print(f"\n\n{root.absolute()}")


if __name__ == "__main__":

    parser = ap.ArgumentParser()
    parser.add_argument("model", choices=utils.MODELS.keys()) 
    parser.add_argument("dataset", choices=utils.DATASETS.keys())
    parser.add_argument("--every", type=int, required=True)
    parser.add_argument("--mods", type=float, nargs='+', default=[0])
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument('--eta', type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--name", type=str, default="")
    parser.add_argument("--deterministic", action='store_true', default=False)
    args = parser.parse_args()

    main(args)

    
