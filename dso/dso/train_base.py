"""Defines main training loop for deep symbolic optimization."""

import os
import json
from itertools import compress
from abc import ABC, abstractmethod

import tensorflow as tf
import numpy as np

from dso.program import Program, from_tokens
from dso.utils import weighted_quantile
from dso.memory import Batch, make_queue
from dso.variance import quantile_variance

# Ignore TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Set TensorFlow seed
tf.set_random_seed(0)

# Work for multiprocessing pool: compute reward
def work(p):
    """Compute reward and return it with optimized constants"""
    r = p.r
    return p

class Trainer(ABC):
    def __init__(self, sess, policy, policy_optimizer, gp_controller, logger,
                 pool, n_samples=2000000, batch_size=1000, alpha=0.5,
                 epsilon=0.05, verbose=True, baseline="R_e",
                 b_jumpstart=False, early_stopping=True, debug=0,
                 use_memory=False, memory_capacity=1e3,  warm_start=None, memory_threshold=None,
                 complexity="token", const_optimizer="scipy", const_params=None,  n_cores_batch=1,
                 n_cores_task=1):

        """
        Initializes the main training loop.

        Parameters
        ----------
        sess : tf.Session
            TensorFlow Session object.
        
        policy : dso.policy.Policy
            Parametrized probability distribution over discrete objects.
            Used to generate programs and compute loglikelihoods.

        policy_optimizer : dso.policy_optimizer.policy_optimizer
            policy_optimizer object used to optimize the policy.

        gp_controller : dso.gp.gp_controller.GPController or None
            GP controller object used to generate Programs.

        logger : dso.train_stats.StatsLogger
            Logger to save results with

        pool : multiprocessing.Pool or None
            Pool to parallelize reward computation. For the control task, each
            worker should have its own TensorFlow model. If None, a Pool will be
            generated if n_cores_batch > 1.

        n_samples : int or None, optional
            Total number of objects to sample. This may be exceeded depending
            on batch size.

        batch_size : int, optional
            Number of sampled expressions per iteration.

        alpha : float, optional
            Coefficient of exponentially-weighted moving average of baseline.

        epsilon : float or None, optional
            Fraction of top expressions used for training. None (or
            equivalently, 1.0) turns off risk-seeking.

        verbose : bool, optional
            Whether to print progress.

        baseline : str, optional
            Type of baseline to use: grad J = (R - b) * grad-log-prob(expression).
            Choices:
            (1) "ewma_R" : b = EWMA(<R>)
            (2) "R_e" : b = R_e
            (3) "ewma_R_e" : b = EWMA(R_e)
            (4) "combined" : b = R_e + EWMA(<R> - R_e)
            In the above, <R> is the sample average _after_ epsilon sub-sampling and
            R_e is the (1-epsilon)-quantile estimate.

        b_jumpstart : bool, optional
            Whether EWMA part of the baseline starts at the average of the first
            iteration. If False, the EWMA starts at 0.0.

        early_stopping : bool, optional
            Whether to stop early if stopping criteria is reached.

        debug : int, optional
            Debug level, also passed to Controller. 0: No debug. 1: Print initial
            parameter means. 2: Print parameter means each step.

        use_memory : bool, optional
            If True, use memory queue for reward quantile estimation.

        memory_capacity : int
            Capacity of memory queue.

        warm_start : int or None
            Number of samples to warm start the memory queue. If None, uses
            batch_size.

        memory_threshold : float or None
            If not None, run quantile variance/bias estimate experiments after
            memory weight exceeds memory_threshold.

        complexity : str, optional
            Not used

        const_optimizer : str or None, optional
            Not used

        const_params : dict, optional
            Not used

        n_cores_batch : int, optional
            Not used

        n_cores_task : int, optional
            Not used

        sync : bool
            Flag to indicate whether training should be run with synchronous
            parallelisation or not.
        """
        self.sess = sess
        # Initialize compute graph
        self.sess.run(tf.global_variables_initializer())

        self.policy = policy
        self.policy_optimizer = policy_optimizer
        self.gp_controller = gp_controller
        self.logger = logger
        self.pool = pool
        self.n_samples = n_samples
        self.batch_size = batch_size
        self.alpha = alpha
        self.epsilon = epsilon
        self.verbose = verbose
        self.baseline = baseline
        self.b_jumpstart = b_jumpstart
        self.early_stopping = early_stopping
        self.debug = debug
        self.use_memory = use_memory
        self.memory_threshold = memory_threshold
        self.n_cores_task = n_cores_task

        if self.debug:
            tvars = tf.trainable_variables()
            def print_var_means():
                tvars_vals = self.sess.run(tvars)
                for var, val in zip(tvars, tvars_vals):
                    print(var.name, "mean:", val.mean(), "var:", val.var())
            self.print_var_means = print_var_means

        self.priority_queue = None

        # Create the memory queue
        if self.use_memory:
            assert self.epsilon is not None and self.epsilon < 1.0, \
                "Memory queue is only used with risk-seeking."
            self.memory_queue = make_queue(policy=self.policy, priority=False,
                                           capacity=int(memory_capacity))

            # Warm start the queue
            # TBD: Parallelize. Abstract sampling a Batch
            warm_start = warm_start if warm_start is not None else self.batch_size
            actions, obs, priors = policy.sample(warm_start)
            programs = [from_tokens(a) for a in actions]
            r = np.array([p.r for p in programs])
            l = np.array([len(p.traversal) for p in programs])
            on_policy = np.array([p.originally_on_policy for p in programs])
            sampled_batch = Batch(actions=actions, obs=obs, priors=priors,
                                  lengths=l, rewards=r, on_policy=on_policy)
            self.memory_queue.push_batch(sampled_batch, programs)
        else:
            self.memory_queue = None

        self.nevals = 0 # Total number of sampled expressions (from RL or GP)
        self.iteration = 0 # Iteration counter
        self.r_best = -np.inf
        self.p_r_best = None
        self.done = False

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
            for p in programs:
                Program.cache[p.str] = p
        
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
    
    def compute_rewards_parallel(self, programs):
        """
        If using a process pool and not synchronous, parallelize the reward
        computation by distributing to self.pool.
        """
        if (self.pool is None) or self.synchronous:
            # No parallel reward or synchronous approach => do nothing
            return programs

        # Filter out programs that have not been evaluated
        programs_to_optimize = list(set([p for p in programs if "r" not in p.__dict__]))
        pool_p_dict = { p.str : p for p in self.pool.map(work, programs_to_optimize) }
        programs = [pool_p_dict[p.str] if "r" not in p.__dict__  else p for p in programs]
        # Make sure to update cache with new programs
        Program.cache.update(pool_p_dict)
        return programs
    
    def risk_seeking_filter(self, programs, rewards):
        """
        Apply an epsilon risk-seeking filter to keep top quantile of programs.
        If using memory-based weighting, do so as well.
        Returns the filtered subset of programs, plus the updated arrays.
        """
        # No risk seeking
        if (self.epsilon is None) or (self.epsilon >= 1.0):
            return programs, rewards, np.ones_like(rewards, dtype=int), None

        # Weighted memory approach
        if self.use_memory:
            unique_programs = [p for p in programs if p.str not in self.memory_queue.unique_items]
            N = len(unique_programs)
            memory_r = self.memory_queue.get_rewards()
            sample_r = [p.r for p in unique_programs]
            combined_r = np.concatenate([memory_r, sample_r])
            memory_w = self.memory_queue.compute_probs()

            if N == 0:
                print("WARNING: Found no unique samples in batch!")
                combined_w = memory_w / memory_w.sum()
            else:
                sample_w = np.repeat((1 - memory_w.sum()) / N, N)
                combined_w = np.concatenate([memory_w, sample_w])

            # Quantile variance/bias estimates
            if self.memory_threshold is not None:
                print("Memory weight:", memory_w.sum())
                if memory_w.sum() > self.memory_threshold:
                    quantile_variance(self.memory_queue, self.policy, self.batch_size, self.epsilon, self.iteration)

            quantile = weighted_quantile(values=combined_r, weights=combined_w, q=1 - self.epsilon)
        else:
            # Empirical quantile
            # The 'higher' interpolation ensures we filter strictly above the quantile
            quantile = np.quantile(rewards, 1 - self.epsilon, interpolation="higher")

        keep = rewards >= quantile
        # Filter
        kept_programs = list(compress(programs, keep))
        kept_rewards = rewards[keep]
        return kept_programs, kept_rewards, keep, quantile

    def save(self, save_path):
        """
        Save the state of the Trainer.
        """

        state_dict = {
            "nevals" : self.nevals,
            "iteration" : self.iteration,
            "r_best" : self.r_best,
            "p_r_best_tokens" : self.p_r_best.tokens.tolist() if self.p_r_best is not None else None
        }
        with open(save_path, 'w') as f:
            json.dump(state_dict, f)

        print("Saved Trainer state to {}.".format(save_path))

    def load(self, load_path):
        """
        Load the state of the Trainer.
        """

        with open(load_path, 'r') as f:
            state_dict = json.load(f)

        # Load nevals and iteration from savestate
        self.nevals = state_dict["nevals"]
        self.iteration = state_dict["iteration"]

        # Load r_best and p_r_best
        if state_dict["p_r_best_tokens"] is not None:
            tokens = np.array(state_dict["p_r_best_tokens"], dtype=np.int32)
            self.p_r_best = from_tokens(tokens)
        else:
            self.p_r_best = None

        print("Loaded Trainer state from {}.".format(load_path))

    @abstractmethod
    def run_one_step(self, override=None):
        """
        Executes one step of main training loop. If override is given,
        train on that batch. Otherwise, sample the batch to train on.

        Parameters
        ----------
        override : tuple or None
            Tuple of (actions, obs, priors, programs) to train on offline
            samples instead of sampled
        """
        raise NotImplementedError