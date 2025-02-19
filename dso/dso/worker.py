import tensorflow as tf
import numpy as np
from itertools import compress
import multiprocessing as mp

from dso.memory import Batch
from dso.policy_optimizer.pg_policy_optimizer import PGPolicyOptimizer
from dso.program import Program, from_tokens
from dso.tf_state_manager import HierarchicalStateManager, HierarchicalStateManager


class Worker(mp.Process):
    def __init__(
            self,
            worker_id,
            policy_class,  # e.g. RNNPolicy
            prior,
            policy_kwargs,  # dict of arguments for your policy constructor
            policy_optimizer_kwargs,
            state_manager_kwargs,
            task_queue,  # receives commands like {"type": "sample", "batch_size": ...}
            result_queue,  # used to send sampled batches (or other data) back
            param_queue,  # used to receive updated parameters from the main process
            batch_size,
            epsilon,
            b_jumpstart,
            baseline,
            alpha,
    ):
        super().__init__()
        # tf.reset_default_graph()
        self.sess = None
        self.policy = None
        self.policy_optimizer = None
        self.state_manager = None

        self.worker_id = worker_id
        self.policy_class = policy_class
        self.prior = prior
        self.policy_kwargs = {key: value for key,value in policy_kwargs.items() if key != 'policy_type'}
        self.policy_optimizer_kwargs = {key:value for key,value in policy_optimizer_kwargs.items() if key != 'policy_optimizer_type'}
        self.state_manager_kwargs = state_manager_kwargs
        self.state_manager_kwargs = {key: value for key,value in state_manager_kwargs.items() if key != 'type'}
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.param_queue = param_queue
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.b_jumpstart = b_jumpstart
        self.baseline = baseline
        self.alpha = alpha

        self.r_best = -np.inf
        self.p_r_best = None

    def run(self):
        config = tf.ConfigProto(device_count={'GPU': 0})
        if self.sess is None:
            self.sess = tf.Session(config=config)
        if self.state_manager is None:
            self.state_manager = HierarchicalStateManager(**self.state_manager_kwargs)

        if self.policy is None:
            self.policy = self.policy_class(
                self.sess,
                self.prior,
                self.state_manager,
                self.worker_id,
                **self.policy_kwargs
            )

        if self.policy_optimizer is None:
            self.policy_optimizer = PGPolicyOptimizer(
                self.sess,
                self.policy,
                **self.policy_optimizer_kwargs
            )

        self.sess.run(tf.global_variables_initializer())

        while True:
            task = self.task_queue.get()
            if task is None:
                print(f"Worker {self.worker_id} stopping...")
                break

            elif task["type"] == "update_params":
                new_params = task["params"]
                self.set_params(new_params)

            elif task["type"] == "sample":
                ewma = None if self.b_jumpstart else 0.0  # EWMA portion of baseline

                override = task["override"]
                actions, obs, priors, programs, n_extra = self.sample_batch(override)
                for p in programs:
                    Program.cache[p.str] = p

                r = np.array([p.r for p in programs])
                r_max = np.max(r)
                l = np.array([len(p.traversal) for p in programs])
                on_policy = np.array([p.originally_on_policy for p in programs])

                programs, r, keep, quantile = self.risk_seeking_filter(programs, r)
                actions = actions[keep, :]
                obs = obs[keep, :, :]
                priors = priors[keep, :, :]
                l = l[keep]
                on_policy = on_policy[keep]

                b, _ = self.compute_baseline(r, quantile, ewma)

                sampled_batch = Batch(actions=actions, obs=obs, priors=priors,
                                      lengths=l,
                                      rewards=r, on_policy=on_policy)

                grads = self.policy_optimizer.compute_grads(b, sampled_batch)

                # Update best expression
                if r_max > self.r_best:
                    self.r_best = r_max
                    self.p_r_best = programs[np.argmax(r)]

                data = {
                    "worker_id": self.worker_id,
                    "grads": grads,
                    "r_best": self.r_best,
                    "p_r_best": self.p_r_best,
                }
                self.result_queue.put(data)

            else:
                raise NotImplementedError(f"Worker {self.worker_id} received unknown task type: {task['type']}")

        self.sess.close()

    def compute_baseline(self, r, quantile, ewma):
        # NOTE: pg_loss = tf.reduce_mean((self.r - self.baseline) * neglogp, name="pg_loss")
        if self.baseline == "ewma_R":
            ewma = np.mean(r) if ewma is None else self.alpha*np.mean(r) + (1 - self.alpha)*ewma
            b = ewma
        elif self.baseline == "R_e": # Default
            ewma = -1
            b = quantile
        elif self.baseline == "ewma_R_e":
            ewma = np.min(r) if ewma is None else self.alpha*quantile + (1 - self.alpha)*ewma
            b = ewma
        elif self.baseline == "combined":
            ewma = np.mean(r) - quantile if ewma is None else self.alpha*(np.mean(r) - quantile) + (1 - self.alpha)*ewma
            b = quantile + ewma
        else:
            raise NotImplementedError('Baseline not recognised.')

        return b, ewma

    def sample_batch(self, override):
        """
        Sample a batch of expressions from the policy, or if override is provided,
        use the override data.
        Returns (actions, obs, priors, programs) for the next training iteration.
        """
        if override is None:
            # Sample batch of Programs from the Controller
            actions, obs, priors = self.policy.sample(self.batch_size)
            programs = [from_tokens(a) for a in actions]
        else:
            # Train on the given batch of Programs
            actions, obs, priors, programs = override

        n_extra = 0

        # Possibly add extended batch
        if self.policy.valid_extended_batch:
            self.policy.valid_extended_batch = False
            n_extra = self.policy.extended_batch[0]
            if n_extra > 0:
                extra_programs = [from_tokens(a) for a in self.policy.extended_batch[1]]
                actions = np.concatenate([actions, self.policy.extended_batch[1]])
                obs = np.concatenate([obs, self.policy.extended_batch[2]])
                priors = np.concatenate([priors, self.policy.extended_batch[3]])
                programs += extra_programs
        return actions, obs, priors, programs, n_extra

    def risk_seeking_filter(self, programs, rewards):
        """
        Apply an epsilon risk-seeking filter to keep top quantile of programs.
        If using memory-based weighting, do so as well.
        Returns the filtered subset of programs, plus the updated arrays.
        """
        # No risk seeking
        if (self.epsilon is None) or (self.epsilon >= 1.0):
            return programs, rewards, np.ones_like(rewards, dtype=int), None

        # Empirical quantile
        # The 'higher' interpolation ensures we filter strictly above the quantile
        quantile = np.quantile(rewards, 1 - self.epsilon, interpolation="higher")

        keep = rewards >= quantile
        # Filter
        kept_programs = list(compress(programs, keep))
        kept_rewards = rewards[keep]
        return kept_programs, kept_rewards, keep, quantile

    def set_params(self, params):
        """
        Set the parameters of the policy according to broadcast
        from main process.
        """
        self.policy.set_params(params, self.worker_id)