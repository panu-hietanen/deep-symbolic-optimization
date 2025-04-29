
import tensorflow as tf
import numpy as np
from dso.policy_optimizer import PolicyOptimizer
from dso.policy import Policy
from dso.memory import Batch

class PPOPolicyOptimizer(PolicyOptimizer):
    """Proximal policy optimization policy optimizer.

    Parameters
    ----------

    ppo_clip_ratio : float
        Clip ratio to use for PPO.

    ppo_n_iters : int
        Number of optimization iterations for PPO.

    ppo_n_mb : int
        Number of minibatches per optimization iteration for PPO.
        
    """
    def __init__(self, 
            sess : tf.Session,
            policy : Policy,
            debug : int = 0, 
            summary : bool = False,
            # Optimizer hyperparameters
            optimizer : str = 'adam',
            learning_rate : float = 0.001,
            # Loss hyperparameters
            entropy_weight : float = 0.005,
            entropy_gamma : float = 1.0,
            # PPO hyperparameters
            ppo_clip_ratio : float = 0.2,
            ppo_n_iters : int = 10,
            ppo_n_mb : int = 4) -> None:
        self.ppo_clip_ratio = ppo_clip_ratio
        self.ppo_n_iters = ppo_n_iters
        self.ppo_n_mb = ppo_n_mb
        self.rng = np.random.RandomState(0) # Used for PPO minibatch sampling
        super()._setup_policy_optimizer(sess, policy, debug, summary, optimizer, learning_rate, entropy_weight, entropy_gamma)


    def _set_loss(self):
        with tf.name_scope("losses"):
            # Retrieve rewards from batch
            r = self.sampled_batch_ph.rewards

            self.old_neglogp_ph = tf.placeholder(dtype=tf.float32, 
                                    shape=(None,), name="old_neglogp")
            ratio = tf.exp(self.old_neglogp_ph - self.neglogp)
            clipped_ratio = tf.clip_by_value(ratio, 1. - self.ppo_clip_ratio,
                                        1. + self.ppo_clip_ratio)
            ppo_loss = -tf.reduce_mean((r - self.baseline) *
                                    tf.minimum(ratio, clipped_ratio))
            # Loss already is set to entropy loss
            self.loss += ppo_loss

            # Define PPO diagnostics
            clipped = tf.logical_or(ratio < (1. - self.ppo_clip_ratio),
                                ratio > 1. + self.ppo_clip_ratio)
            self.clip_fraction = tf.reduce_mean(tf.cast(clipped, tf.float32))
            self.sample_kl = tf.reduce_mean(self.neglogp - self.old_neglogp_ph)


    def _preppend_to_summary(self):
        with tf.name_scope("summary"):
            tf.summary.scalar("ppo_loss", self.ppo_loss)

    def compute_grads(self, baseline, sampled_batch):
        """
        Compute—but do not apply—PPO gradients for the given baseline & Batch.
        Returns a list of length `ppo_n_iters`, each element is itself
        a list of `ppo_n_mb` lists-of-gradient‐arrays.
        """
        # 1) Prepare the feed_dict for the full batch
        feed_dict = {
            self.baseline: baseline,
            self.sampled_batch_ph: sampled_batch
        }
        n = sampled_batch.rewards.shape[0]

        # 2) Compute old_neglogp for _all_ samples
        old_neglogp = self.sess.run(self.neglogp, feed_dict=feed_dict)

        # 3) Now do ppo_n_iters passes over shuffled minibatches,
        #    but instead of sess.run(train_op) we run self.grads
        all_grad_lists = []
        idx = np.arange(n)
        for _ in range(self.ppo_n_iters):
            self.rng.shuffle(idx)
            mb_indices = np.array_split(idx, self.ppo_n_mb)

            iter_grad_lists = []
            for mb in mb_indices:
                # slice out a smaller Batch for this minibatch
                mb_batch = Batch(
                    **{k: v[mb] for k, v in sampled_batch._asdict().items()}
                )
                mb_feed = {
                        self.baseline: baseline,
                        self.batch_size: len(mb),
                        self.old_neglogp_ph: old_neglogp[mb],
                        self.sampled_batch_ph: mb_batch
                }
                # 4) RUN the gradient tensors, *not* the train_op
                grad_list = self.sess.run(self.grads, feed_dict=mb_feed)
                iter_grad_lists.append(grad_list)

            all_grad_lists.append(iter_grad_lists)

        # all_grad_lists is shape (ppo_n_iters, ppo_n_mb, [ num_params ])
        return all_grad_lists

    def apply_grads(self, grad_list):
        """Apply external gradients."""
        feed_dict = {}
        for placeholder, array in zip(self.grad_placeholders, grad_list):
            feed_dict[placeholder] = array
        self.sess.run(self.apply_op, feed_dict=feed_dict)

    def train_step(self, baseline, sampled_batch):
        feed_dict = {
            self.baseline : baseline,
            self.sampled_batch_ph : sampled_batch
        }
        n_samples = sampled_batch.rewards.shape[0]


        # Compute old_neglogp to be used for training
        old_neglogp = self.sess.run(self.neglogp, feed_dict=feed_dict)

        # Perform multiple steps of minibatch training
        # feed_dict[self.old_neglogp_ph] = old_neglogp
        indices = np.arange(n_samples)
        for ppo_iter in range(self.ppo_n_iters):
            self.rng.shuffle(indices) # in-place
            # list of [ppo_n_mb] arrays
            minibatches = np.array_split(indices, self.ppo_n_mb)
            for i, mb in enumerate(minibatches):
                sampled_batch_mb = Batch(
                        **{name: array[mb] for name, array
                           in sampled_batch._asdict().items()})
                mb_feed_dict = {
                        self.baseline: baseline,
                        self.batch_size: len(mb),
                        self.old_neglogp_ph: old_neglogp[mb],
                        self.sampled_batch_ph: sampled_batch_mb
                }

                summaries, _ = self.sess.run([self.summaries, self.train_op],
                                                 feed_dict=mb_feed_dict)

                # Diagnostics
                # kl, cf, _ = self.sess.run(
                #     [self.sample_kl, self.clip_fraction, self.train_op],
                #     feed_dict=mb_feed_dict)
                # print("ppo_iter", ppo_iter, "i", i, "KL", kl, "CF", cf)

        return summaries




