"""Parallelized, single-point launch script to run DSO on a set of benchmarks."""

import os
import time
import commentjson as json
import numpy as np
import pandas as pd
import copy
import itertools

from run_utils import run_experiment, clean_config

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

def benchmark(config, benchmarks, runs=1):
    summaries = []
    timestamp = None
    print(f"INFO: RUNNING {len(benchmarks)} BENCHMARKS {runs} TIMES")
    for i, benchmark in enumerate(benchmarks):
        print(f"\n=== Dataset {benchmark} ===")
        config_mod = copy.deepcopy(config)

        exp_suffix = benchmark

        config_mod["task"]["dataset"] = benchmark

        config_mod, runs, n_cores_task = clean_config(config_mod, runs=runs)
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

        start = time.time()
        summary_path = run_experiment(config_mod, runs, n_cores_task)
        end = time.time()

        summary = pd.read_csv(summary_path)

        summary["dataset"] = benchmark
        summaries.append(summary)
        print(f"=== FINISHED BENCHMARK {benchmark} IN {end - start: .4f} SECONDS===")
        print(summary)

    return summaries, timestamp

def postprocess(summaries, timestamp, save_results=False):
    all_results = pd.concat(summaries, ignore_index=True)
    all_results_sorted = all_results.sort_values(by=["dataset", "t"], ascending=[True, True])

    grouped = all_results_sorted.groupby("dataset")

    summary_df = grouped.agg(
        success_rate=("success", "mean"),
        avg_time=("t", "mean"),
        total_runs=("success", "count"),
        success_count=("success", "sum"),
    ).reset_index()

    summary_df["failure_count"] = summary_df["total_runs"] - summary_df["success_count"]
    summary_df["success_rate"] = 100.0 * summary_df["success_rate"]
    summary_df["std_time"] = grouped["t"].std().values
    summary_df["min_time"] = grouped["t"].min().values
    summary_df["max_time"] = grouped["t"].max().values

    print("== RESULTS ==")
    print(summary_df)
    if save_results:
        folder = f'./log/bench_{timestamp}'
        os.makedirs(folder, exist_ok=True)
        print(f"Saving results to {folder}...")
        all_results_sorted.to_csv(f'{folder}/results.csv', index=False)
        summary_df.to_csv(f'{folder}/summary.csv', index=False)

def main(save_results=False, config_path='', runs=1):
    try:
        with open(config_path, encoding='utf-8') as f:
            config = json.load(f)
    except Exception as e:
        raise ValueError(f'Error reading config file {config_path}: {e}')

    # Benchmarks
    # benchmarks = [f'Nguyen-{i}' for i in range(1,13)]
    benchmarks = ['Nguyen-1']

    start = time.time()
    summaries, timestamp = benchmark(config, benchmarks, runs)
    end = time.time()
    print(f"Time taken to run search: {end - start: .4f} seconds")

    postprocess(summaries, timestamp, save_results)

if __name__ == "__main__":
    save_results = True
    config_path = '/homes/55/panu/4yp/deep-symbolic-optimization/dso/dso/config/config_regression.json'
    runs = 2
    main(save_results, config_path, runs)

