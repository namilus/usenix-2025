import tensorflow as tf
import numpy as np

class ThudiForgingAttack:
    def __init__(self, x_data, y_data, error_fn, seed=2024, M=1):
        self.x_data = x_data
        self.y_data = y_data
        self.error_fn = error_fn
        self.M = M
        self.rng = np.random.default_rng(seed=seed)

    def forge(self, x, y, true_idx, num_to_forge):
        if num_to_forge == 0:
            return x, y

        possible_indices = np.setdiff1d(np.arange(len(self.x_data)), true_idx)
        M_batch_indices = [self.rng.choice(possible_indices, num_to_forge) for _ in range(self.M)]
        if num_to_forge == x.shape[0]:
            M_batches_x = [tf.gather(self.x_data, mb) for mb in M_batch_indices]
            M_batches_y = [tf.gather(self.y_data, mb) for mb in M_batch_indices]
        else:
            M_batches_x = [tf.concat([tf.gather(self.x_data, mb),
                                      x[num_to_forge:]], axis=0) for mb in M_batch_indices]
            M_batches_y = [tf.concat([tf.gather(self.y_data, mb),
                                      y[num_to_forge:]], axis=0) for mb in M_batch_indices]
            

        grad_errors = tf.stack([self.error_fn(x,y) for (x,y) in zip(M_batches_x, M_batches_y)], axis=0)
        best = tf.argmin(grad_errors)
        return M_batches_x[best], M_batches_y[best]
        
        
class ZhangForgingAttack:
    def __init__(self, x_data, y_data, memoised_path):
        self.x_data = x_data
        self.y_data = y_data
        self.memoised_nns = np.load(memoised_path)

    def forge(self, x, y, true_idx, num_to_forge):
        if num_to_forge == 0:
            return x, y

        forging_idx = true_idx[:num_to_forge]
        nn_idx = tf.gather(self.memoised_nns, forging_idx)

        new_indices = []
        for idx in nn_idx:
            for i in idx[1:]:
                if i not in true_idx:
                    new_indices.append(i)
                    break

        # IPython.embed()

        new_x = tf.gather(self.x_data, new_indices)
        new_y = tf.gather(self.y_data, new_indices)

        forged_x = tf.concat([new_x, x[num_to_forge:]], axis=0)
        forged_y = tf.concat([new_y, y[num_to_forge:]], axis=0)

        return forged_x, forged_y

        

