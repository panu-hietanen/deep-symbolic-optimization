"""Parallelized, single-point launch script to run DSO on a set of benchmarks."""

import os
import sys
import time
import multiprocessing
from copy import deepcopy
from datetime import datetime
import commentjson as json
import numpy as np
import pandas as pd
import copy
import itertools

from dso import DeepSymbolicOptimizer
from dso.logeval import LogEval
from dso.config import load_config
from dso.utils import safe_update_summary

from run_utils import print_summary, clean_config, run_experiment

PARAM_MAPPING = {
    'lr': 'learning_rate',
    'ew': 'entropy_weight',
    'eg': 'entropy_gamma',
    'batch_size': 'batch_size',
    'epsilon': 'epsilon',
    'alpha_train': 'alpha',
    'clip': 'ppo_clip_ratio',
    'iters': 'ppo_n_iters',
    'mb': 'ppo_n_mb'
}

CONFIG_MAPPING = {
    'lr': 'policy_optimizer',
    'ew': 'policy_optimizer',
    'eg': 'policy_optimizer',
    'clip': 'policy_optimizer',
    'iters': 'policy_optimizer',
    'mb': 'policy_optimizer',
    'batch_size': 'training',
    'epsilon': 'training',
    'alpha_train': 'training',
}

def benchmark(config, benchmarks, runs=1, n_cores_task=1):
    summaries = []
    cached = []
    timestamp = None
    print("Starting workers...")

    experiments = []
    paths = []

    for i, benchmark in enumerate(benchmarks):
        config_mod = copy.deepcopy(config)

        exp_suffix = benchmark

        config_mod, runs, n_cores_task, messages = clean_config(config_mod, runs=runs, n_cores_task=n_cores_task, seed=i, benchmark=benchmark, time=i)
        # Adjust run directory to keep results separate
        # e.g. append a suffix with the hyperparams
        # Here we incorporate them into the 'exp_name'

        timestamp = config_mod["experiment"]["timestamp"]

        if config_mod["experiment"].get("exp_name") is not None:
            config_mod["experiment"]["exp_name"] += "_" + exp_suffix
        else:
            config_mod["experiment"]["exp_name"] = exp_suffix

        config_mod["experiment"]["exp_name"] += "_" + timestamp
        config_mod["experiment"]["logdir"] = "./log_hypers"

        model = DeepSymbolicOptimizer(deepcopy(config_mod))

        experiment = {
            "config_mod": config_mod,
            "model": model,
            "runs": runs,
            "n_cores_task": n_cores_task,
            "benchmark": benchmark,
            "messages": messages
        }

        experiments.append(experiment)

    print("Beginning experiments.")
    try:
        for i, experiment in enumerate(experiments):

            print(f"\n@@@ Dataset {experiment['benchmark']} @@@")

            print_summary(experiment["config_mod"], experiment["runs"], experiment["messages"])

            start = time.time()
            summary_path, output_prefix = run_experiment(experiment["config_mod"], experiment["runs"], experiment["n_cores_task"], experiment["model"])
            end = time.time()

            paths.append((summary_path, output_prefix))
            summary = pd.read_csv(summary_path)
            if experiment["config_mod"]["logging"]["save_cache"]:
                cache_file = output_prefix + "_cache.csv"
                try:
                    cache = pd.read_csv(cache_file)
                    cached.append(cache)
                except FileNotFoundError:
                    print('Warning: Cache file not found.')
            summary["dataset"] = experiment['benchmark']
            summary["sync"] = experiment["config_mod"]["training"]["sync"]
            summaries.append(summary)
            print(f"@@@ FINISHED BENCHMARK {experiment['benchmark']} IN {end - start: .4f} SECONDS @@@")
            print(summary)
    except KeyboardInterrupt:
        print("Interrupted by user. Saving...")
    except Exception as e:
        print(f"Error {type(e).__name__}: {e}. Trying to recover...")
        summaries = []
        timestamp = "RECOVERY"
        for path, experiment in zip(paths[:-1], experiments[:-1]):
            summary = pd.read_csv(path)

            summary["dataset"] = experiment['benchmark']
            summary["sync"] = experiment["config_mod"]["training"]["sync"]
            summaries.append(summary)


    return summaries, cached, timestamp

def postprocess(summaries, cached, timestamp, save_results=False):
    all_results = pd.concat(summaries, keys=range(len(summaries)))
    all_results.index = all_results.index.droplevel(1)
    all_results_sorted = all_results.sort_values(by=["dataset", "t"], ascending=[True, True])

    grouped = all_results_sorted.groupby("dataset")

    summary_df = grouped.agg(
        success_rate=("success", "mean"),
        avg_time=("t", "mean"),
        total_runs=("success", "count"),
        success_count=("success", "sum"),
        avg_samples = ("n_samples", "mean"),
        avg_reward=("r", "mean"),
    ).reset_index()

    summary_df["failure_count"] = summary_df["total_runs"] - summary_df["success_count"]
    summary_df["success_rate"] = 100.0 * summary_df["success_rate"]

    summary_df["std_reward"] = grouped["r"].std().values
    summary_df["min_reward"] = grouped["r"].min().values
    summary_df["max_reward"] = grouped["r"].max().values

    summary_df["std_time"] = grouped["t"].std().values
    summary_df["min_time"] = grouped["t"].min().values
    summary_df["max_time"] = grouped["t"].max().values

    summary_df["std_samples"] = grouped["n_samples"].std().values
    summary_df["min_samples"] = grouped["n_samples"].min().values
    summary_df["max_samples"] = grouped["n_samples"].max().values

    summary_df["mean_nmse"] = grouped["nmse_test"].mean().values
    summary_df["std_nmse"] = grouped["nmse_test"].std().values
    summary_df["mean_nmse_noiseless"] = grouped["nmse_test_noiseless"].mean().values
    summary_df["std_nmse_noiseless"] = grouped["nmse_test_noiseless"].std().values

    # Filter to successful runs
    successful = all_results_sorted[all_results_sorted["success"] == 1]
    grouped_success = successful.groupby("dataset")

    # Compute stats over only successful runs
    summary_df["avg_time_successful"] = grouped_success["t"].mean().reindex(summary_df["dataset"]).values
    summary_df["std_time_successful"] = grouped_success["t"].std().reindex(summary_df["dataset"]).values

    summary_df["mean_nmse_successful"] = grouped_success["nmse_test"].mean().reindex(summary_df["dataset"]).values
    summary_df["std_nmse_successful"] = grouped_success["nmse_test"].std().reindex(summary_df["dataset"]).values

    summary_df["mean_samples_successful"] = grouped_success["n_samples"].mean().reindex(summary_df["dataset"]).values
    summary_df["std_samples_successful"] = grouped_success["n_samples"].std().reindex(summary_df["dataset"]).values

    if cached:
        all_caches = pd.concat(cached)
        all_caches_sorted = all_caches.sort_values(by="r", ascending=False)
    else:
        all_caches_sorted = None

    print("== RESULTS ==")
    print(summary_df)
    if save_results:
        folder = f'./log/bench_{timestamp}'
        os.makedirs(folder, exist_ok=True)
        print(f"Saving results to {folder}...")
        all_results_sorted.to_csv(f'{folder}/results.csv', index=False)
        summary_df.to_csv(f'{folder}/summary.csv', index=False)
        if all_caches_sorted is not None:
            all_caches_sorted.to_csv(f'{folder}/cache.csv', index=False)

def main(save_results=False, config_path='', runs=1, n_cores_task=1):
    try:
        with open(config_path, encoding='utf-8') as f:
            config = json.load(f)
    except Exception as e:
        raise ValueError(f'Error reading config file {config_path}: {e}')

    # Benchmarks
    # benchmarks = [f'Nguyen-{i}' for i in range(1,13)]
    benchmarks = ['Nguyen-1']
    print(f"INFO: RUNNING {len(benchmarks)} BENCHMARKS {runs} TIMES")
    benchmarks *= runs

    start = time.time()
    summaries, cached, timestamp = benchmark(config, benchmarks, n_cores_task=n_cores_task)
    end = time.time()
    print(f"Time taken to run search: {end - start: .4f} seconds")

    postprocess(summaries, cached, timestamp, save_results)

if __name__ == "__main__":
    save_results = True
    config_path = '/homes/55/panu/4yp/deep-symbolic-optimization/dso/dso/config/config_regression.json'
    runs = 2
    n_cores_task = 5
    main(save_results, config_path, runs, n_cores_task)

