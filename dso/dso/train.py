"""Defines main training loop for deep symbolic optimization."""

import os
import time
from itertools import compress

import multiprocessing as mp
import tensorflow as tf
import numpy as np

from dso.program import Program
from dso.utils import empirical_entropy, get_duration, merge_batches
from dso.memory import Batch
from dso.train_base import Trainer
from dso.worker import Worker
from dso.program import Program, from_tokens

# Work for multiprocessing pool: compute reward
def work(p):
    """Compute reward and return it with optimized constants"""
    r = p.r
    return p

class SingleTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def run_one_step(self, override=None):
        positional_entropy = None
        top_samples_per_batch = list()
        if self.debug >= 1:
            print("\nDEBUG: Policy parameter means:")
            self.print_var_means()

        ewma = None if self.b_jumpstart else 0.0 # EWMA portion of baseline

        start_time = time.time()
        if self.verbose:
            print("-- RUNNING ITERATIONS START -------------")


        # Number of extra samples generated during attempt to get
        # batch_size new samples
        n_extra = 0
        # Record previous cache before new samples are added by from_tokens
        s_history = list(Program.cache.keys())

        # Construct the actions, obs, priors, and programs
        # Shape of actions: (batch_size, max_length)
        # Shape of obs: (batch_size, obs_dim, max_length)
        # Shape of priors: (batch_size, max_length, n_choices)
        actions, obs, priors, programs, n_extra = self.sample_batch(override)

        self.nevals += self.batch_size + n_extra

        # Compute rewards in parallel
        programs = self.compute_rewards_parallel(programs)

        # Compute rewards (or retrieve cached rewards)
        r = np.array([p.r for p in programs])

        # Back up programs to save them properly later
        controller_programs = programs.copy() if self.logger.save_token_count else None

        # Need for Vanilla Policy Gradient (epsilon = null)
        l           = np.array([len(p.traversal) for p in programs])
        s           = [p.str for p in programs] # Str representations of Programs
        on_policy   = np.array([p.originally_on_policy for p in programs])
        invalid     = np.array([p.invalid for p in programs], dtype=bool)

        if self.logger.save_positional_entropy:
            positional_entropy = np.apply_along_axis(empirical_entropy, 0, actions)

        if self.logger.save_top_samples_per_batch > 0:
            # sort in descending order: larger rewards -> better solutions
            sorted_idx = np.argsort(r)[::-1]
            top_perc = int(len(programs) * float(self.logger.save_top_samples_per_batch))
            for idx in sorted_idx[:top_perc]:
                top_samples_per_batch.append([self.iteration, r[idx], repr(programs[idx])])

        # Store in variables the values for the whole batch (those variables will be modified below)
        r_full = r
        l_full = l
        s_full = s
        actions_full = actions
        invalid_full = invalid
        r_max = np.max(r)

        """
        Apply risk-seeking policy gradient: compute the empirical quantile of
        rewards and filter out programs with lesser reward.
        """
        programs, r, keep, quantile = self.risk_seeking_filter(programs, r)

        # Filter quantities whose reward >= quantile
        l = l[keep]
        s = list(compress(s, keep))
        invalid = invalid[keep]
        actions = actions[keep, :]
        obs = obs[keep, :, :]
        priors = priors[keep, :, :]
        on_policy = on_policy[keep]

        # Clip bounds of rewards to prevent NaNs in gradient descent
        r = np.clip(r, -1e6, 1e6)

        # Compute baseline
        b, ewma = self.compute_baseline(r, quantile, ewma)

        # Compute sequence lengths
        lengths = np.array([min(len(p.traversal), self.policy.max_length)
                            for p in programs], dtype=np.int32)

        # Create the Batch
        sampled_batch = Batch(actions=actions, obs=obs, priors=priors,
                            lengths=lengths, rewards=r, on_policy=on_policy)

        pqt_batch = None
        # Train the policy
        summaries = self.policy_optimizer.train_step(b, sampled_batch)

        # Walltime calculation for the iteration
        iteration_walltime = time.time() - start_time

        # Update the memory queue
        if self.memory_queue is not None:
            self.memory_queue.push_batch(sampled_batch, programs)

        # Update new best expression
        if r_max > self.r_best:
            self.r_best = r_max
            self.p_r_best = programs[np.argmax(r)]

            # Print new best expression
            if self.verbose or self.debug:
                print("[{}] Training iteration {}, current best R: {:.4f}".format(get_duration(start_time), self.iteration + 1, self.r_best))
                print("\n\t** New best")
                self.p_r_best.print_stats()

        # Collect sub-batch statistics and write output
        self.logger.save_stats(r_full, l_full, actions_full, s_full,
                               invalid_full, r, l, actions, s, s_history,
                               invalid, self.r_best, r_max, ewma, summaries,
                               self.iteration, b, iteration_walltime,
                               self.nevals, controller_programs,
                               positional_entropy, top_samples_per_batch)


        # Stop if early stopping criteria is met
        if self.early_stopping and self.p_r_best.evaluate.get("success"):
            print("[{}] Early stopping criteria met; breaking early.".format(get_duration(start_time)))
            self.done = True

        if self.verbose and (self.iteration + 1) % 10 == 0:
            print("[{}] Training iteration {}, current best R: {:.4f}".format(get_duration(start_time), self.iteration + 1, self.r_best))

        if self.debug >= 2:
            print("\nParameter means after iteration {}:".format(self.iteration + 1))
            self.print_var_means()

        if self.nevals >= self.n_samples:
            self.done = True

        # Increment the iteration counter
        self.iteration += 1

    def compute_rewards_parallel(self, programs):
        """
        If using a process pool and not synchronous, parallelize the reward
        computation by distributing to self.pool.
        """
        if self.pool is None:
            # No parallel reward or synchronous approach => do nothing
            return programs

        # Filter out programs that have not been evaluated
        programs_to_optimize = list(set([p for p in programs if "r" not in p.__dict__]))
        pool_p_dict = { p.str : p for p in self.pool.map(work, programs_to_optimize) }
        programs = [pool_p_dict[p.str] if "r" not in p.__dict__  else p for p in programs]
        # Make sure to update cache with new programs
        Program.cache.update(pool_p_dict)
        return programs

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

class SyncTrainer(Trainer):
    def __init__(
        self,
        sess,
        policy,
        policy_optimizer,
        gp_controller,
        logger,
        pool,
        workers,
        task_queue,
        result_queue,
        param_queue,
        **kwargs
    ):
        super().__init__(
            sess,
            policy,
            policy_optimizer,
            gp_controller,
            logger,
            pool,
            **kwargs
        )
        # Parallel processing setup
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.param_queue = param_queue
        self.workers = workers

    def run_one_step(self, override=None):
        if self.debug >= 1:
            print("\nDEBUG: Policy parameter means:")
            self.print_var_means()

        start_time = time.time()
        if self.verbose:
            print("-- SYNC TRAINING STEP START --")

        params = self.get_params()
        for _ in range(self.n_cores_task):
            self.task_queue.put({"type": "update_params", "params": params})

        # Request batch samples from workers
        for _ in range(self.n_cores_task):
            self.task_queue.put({"type": "sample", "override": override})
        
        # Collect batch samples from workers
        data = [self.result_queue.get() for _ in range(self.n_cores_task)]
        grads = [w["grads"] for w in data]
        r_bests = [w["r_best"] for w in data]
        p_r_bests = [w["p_r_best"] for w in data]
        n_extra = sum([w["n_extra"] for w in data])

        grads = self.accumulate_grads(grads)

        r_max = max(r_bests)

        self.nevals += self.batch_size * self.n_cores_task + n_extra

        # Update best expression
        if r_max > self.r_best:
                self.r_best = r_max
                i = r_bests.index(r_max)
                self.p_r_best = p_r_bests[i]

        # Train the policy
        _ = self.policy_optimizer.apply_grads(grads)

        # Logging
        iteration_walltime = time.time() - start_time
        # self.logger.save_stats(_, _, _, _,
        #                        _, _, _, _, _, s_history,
        #                        _, self.r_best, r_max, ewma, summaries,
        #                        self.iteration, _, iteration_walltime,
        #                        _, _,
        #                        positional_entropy, top_samples_per_batch)

        # Stop if early stopping criteria is met
        if self.early_stopping and self.p_r_best.evaluate.get("success"):
            print("[{}] Early stopping criteria met; breaking early.".format(get_duration(start_time)))
            self.done = True

        if self.verbose and (self.iteration + 1) % 10 == 0:
            print("[{}] Training iteration {}, current best R: {:.4f}".format(get_duration(start_time), self.iteration + 1, self.r_best))

        if self.debug >= 2:
            print("\nParameter means after iteration {}:".format(self.iteration + 1))
            self.print_var_means()

        if self.nevals >= self.n_samples:
            self.done = True

        # Increment the iteration counter
        if self.done:
            for _ in range(self.n_cores_task):
                self.task_queue.put(None)

            caches = [self.result_queue.get() for _ in range(self.n_cores_task)]
            for cache in caches:
                Program.cache.update(cache)
        self.iteration += 1

    def accumulate_grads(self, grads_list):
        n_arrays = len(grads_list[0])
        result = np.array([np.zeros_like(arr, dtype=float) for arr in grads_list[0]], dtype=object)

        for g in grads_list:
            for i in range(n_arrays):
                result[i] += g[i]

        result /= self.n_cores_task

        return tuple(i for i in result)

    def get_params(self):
        """Get the current parameters of the policy"""
        return self.policy.get_params_numpy(0)