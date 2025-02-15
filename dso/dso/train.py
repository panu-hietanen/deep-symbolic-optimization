"""Defines main training loop for deep symbolic optimization."""

import os
import time
from itertools import compress

import multiprocessing as mp
import tensorflow as tf
import numpy as np

from dso.program import Program
from dso.utils import empirical_entropy, get_duration
from dso.memory import Batch
from dso.train_base import Trainer

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

class SyncTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Parallel processing setup
        self.task_queue = mp.Queue()
        self.result_queue = mp.Queue()
        self.barrier = mp.Barrier(self.n_cores_task + 1)  # +1 for main process

        self.workers = [mp.Process(target=self.worker_function, args=(self.task_queue, self.result_queue, self.barrier))
                        for _ in range(self.n_cores_task)]
        for w in self.workers:
            w.start()

    def worker_function(self, task_queue, result_queue, barrier):
        """Persistent worker that listens for batch sampling or reward computation tasks."""
        while True:
            try:
                task = task_queue.get()
                if task is None:
                    break  # Stop signal
                elif task["type"] == "sample":
                    override = task.get("override", None)
                    actions, obs, priors, programs, n_extra = self.sample_batch(override)
                    result_queue.put((actions, obs, priors, programs, n_extra))
                elif task["type"] == "reward":
                    programs = task["programs"]
                    for p in programs:
                        p.r  # Force reward computation
                    result_queue.put(programs)
                barrier.wait(timeout=10)
            except Exception as e:
                print(f"Worker encountered error: {e}")
                break

    def run_one_step(self, override=None):
        s_history = list(Program.cache.keys())
        positional_entropy = None
        top_samples_per_batch = list()
        if self.debug >= 1:
            print("\nDEBUG: Policy parameter means:")
            self.print_var_means()

        ewma = None if self.b_jumpstart else 0.0 # EWMA portion of baseline

        start_time = time.time()
        if self.verbose:
            print("-- SYNC TRAINING STEP START --")

        # Request batch samples from workers
        for _ in range(self.n_cores_task):
            self.task_queue.put({"type": "sample", "override": override})
        
        # Collect batch samples from workers
        batches = [self.result_queue.get() for _ in range(self.n_cores_task)]
        n_extra = sum([n for _, _, _, _, n in batches])
        batches = [(b[0], b[1], b[2], b[3]) for b in batches]

        actions, obs, priors, programs = self.merge_batches(batches)

        self.nevals += self.batch_size + n_extra

        # Request reward computation from workers
        programs_split = np.array_split(programs, self.n_cores_task)
        for batch in programs_split:
            self.task_queue.put({"type": "reward", "programs": batch})
        
        # Collect computed rewards
        programs = sum([self.result_queue.get() for _ in range(self.n_cores_task)], [])
        r = np.array([p.r for p in programs])
        r_max = np.max(r)

        # Need for Vanilla Policy Gradient (epsilon = null)
        l           = np.array([len(p.traversal) for p in programs])
        s           = [p.str for p in programs] # Str representations of Programs
        on_policy   = np.array([p.originally_on_policy for p in programs])
        invalid     = np.array([p.invalid for p in programs], dtype=bool)

        # Logging
        if self.logger.save_positional_entropy:
            positional_entropy = np.apply_along_axis(empirical_entropy, 0, actions)

        if self.logger.save_top_samples_per_batch > 0:
            # sort in descending order: larger rewards -> better solutions
            sorted_idx = np.argsort(r)[::-1]
            top_perc = int(len(programs) * float(self.logger.save_top_samples_per_batch))
            for idx in sorted_idx[:top_perc]:
                top_samples_per_batch.append([self.iteration, r[idx], repr(programs[idx])])

        # Back up programs for logging
        controller_programs = programs.copy() if self.logger.save_token_count else None

        # Store for logging
        r_full = r
        l_full = l
        s_full = s
        actions_full = actions
        invalid_full = invalid
        r_max = np.max(r)

        # Compute risk-seeking filter
        programs, r, keep, quantile = self.risk_seeking_filter(programs, r)
        actions = actions[keep, :]
        obs = obs[keep, :, :]
        priors = priors[keep, :, :]

        l = l[keep]
        s = list(compress(s, keep))
        invalid = invalid[keep]
        on_policy = on_policy[keep]

        # Compute baseline
        b, ewma = self.compute_baseline(r, quantile, ewma)

        # Create the Batch
        sampled_batch = Batch(actions=actions, obs=obs, priors=priors,
                              lengths=l,
                              rewards=r, on_policy=on_policy)

        # Train the policy
        summaries = self.policy_optimizer.train_step(b, sampled_batch)

        # Update memory queue
        if self.memory_queue is not None:
            self.memory_queue.push_batch(sampled_batch, programs)

        # Update best expression
        if r_max > self.r_best:
            self.r_best = r_max
            self.p_r_best = programs[np.argmax(r)]

        # Logging
        iteration_walltime = time.time() - start_time
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

    def merge_batches(self, worker_batches):
        """Merge batches received from worker processes."""
        all_actions, all_obs, all_priors, all_programs = zip(*worker_batches)
        return (np.concatenate(all_actions, axis=0),
                np.concatenate(all_obs, axis=0),
                np.concatenate(all_priors, axis=0),
                sum(all_programs, []))

    def stop_workers(self):
        """Stop all worker processes cleanly."""
        for _ in self.workers:
            self.task_queue.put(None)
        for w in self.workers:
            w.join()
        self.task_queue.close()
        self.result_queue.close()