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
    'mb': 'ppo_n_mb',
    'bench': 'dataset'
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
    'bench': 'task'
}

def grid_search(config, param_dicts, n_cores_task):
    summaries = []
    timestamp = None
    print(f"INFO: RUNNING {len(param_dicts)} EXPERIMENTS")
    print("Starting workers...")

    experiments = []

    for i, params in enumerate(param_dicts):
        config_mod = copy.deepcopy(config)

        exp_suffix = ""
        for param in params:
            config_mod[CONFIG_MAPPING[param]][PARAM_MAPPING[param]] = params[param]
            exp_suffix += f"{param}-{params[param]}_"
            if param in ['clip', 'iters', 'mb']:
                config_mod['policy_optimizer']['policy_optimizer_type'] = 'ppo'
        exp_suffix = exp_suffix[:-1]

        config_mod, runs, n_cores_task, messages = clean_config(config_mod, n_cores_task=n_cores_task, seed=i, time=i)
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
            "params": json.dumps(params),
            "messages": messages
        }

        experiments.append(experiment)

    print("Beginning experiments.")
    try:
        for i, experiment in enumerate(experiments):

            print(f"\n@@@ Running grid search with {experiment['params']} @@@")

            print_summary(experiment["config_mod"], experiment["runs"], experiment["messages"])

            summary_path = run_experiment(experiment["config_mod"], experiment["runs"], experiment["n_cores_task"], experiment["model"])

            parameters = json.dumps(experiment["params"])
            summary = pd.read_csv(summary_path)

            summary["params_json"] = parameters
            summaries.append(summary)
            try:
                t = float(summary["t"])
            except TypeError:
                print("Warning: Summary not in expected format.")
                t = float(summary["t"].min())
            print(f"@@@ FINISHED ITERATION {i} in {t: .4f} seconds @@@")
            print(summary)
    except KeyboardInterrupt:
        print("Interrupted by user. Saving...")
        return summaries, timestamp

    return summaries, timestamp

def postprocess(summaries, timestamp, save_results=False):
    all_results = pd.concat(summaries, keys=range(len(summaries)))
    all_results.index = all_results.index.droplevel(1)
    all_results_sorted = all_results.sort_values(by="t", ascending=True)

    print(all_results_sorted)
    if save_results:
        folder = f'./log/hypers_{timestamp}'
        os.makedirs(folder, exist_ok=True)
        print(f"Saving results to {folder}...")
        all_results_sorted.to_csv(f'{folder}/results.csv', index=True)

def main(save_results=False, config_path='', random=False, trials=None, n_cores_task=1):
    try:
        with open(config_path, encoding='utf-8') as f:
            config = json.load(f)
    except Exception as e:
        raise ValueError(f'Error reading config file {config_path}: {e}')

    # Training Parameters
    batch_sizes = [500, 1000, 5000]
    epsilons = [0.01, 0.05, 0.1]

    # Vanilla PG Parameters
    learning_rates = [5e-6, 5e-5, 5e-4]
    entropy_weights = [0.01, 0.03, 0.1]
    entropy_gammas = [0.5, 0.75, 0.99]

    # PPO Parameters
    ppo_clip_ratio  = [0.1, 0.2, 0.3]
    ppo_n_iters = [5, 10, 15]
    ppo_n_mb = [1, 4, 8]

    # Benchmarks
    benchmarks = ['Nguyen-2', 'Nguyen-1']

    param_dicts = [
        {"lr": lr, "ew": ew, "eg": eg, "batch_size": bs, "epsilon": ep}
        for lr, ew, eg, bs, ep in
        itertools.product(learning_rates, entropy_weights, entropy_gammas, batch_sizes, epsilons)
    ]

    # param_dicts = [
    #     {"lr": lr, "bench": b}
    #     for lr, b in itertools.product(learning_rates, benchmarks)
    # ]

    # For testing
    # param_dicts = [
    #     {"lr": lr}
    #     for lr in learning_rates
    # ]

    start = time.time()
    if random:
        if trials is None:
            raise ValueError("Must provide trials when random is True.")
        param_dicts = np.random.choice(param_dicts, trials, replace=False)
        summaries, timestamp = grid_search(config, param_dicts, n_cores_task)
    else:
        summaries, timestamp = grid_search(config, param_dicts, n_cores_task)
    end = time.time()
    print(f"Time taken to run search: {end - start: .4f} seconds")

    postprocess(summaries, timestamp, save_results)

if __name__ == "__main__":
    save_results = True
    config_path = '/homes/55/panu/4yp/deep-symbolic-optimization/dso/dso/config/config_regression.json'
    random = False
    trials = 10
    n_cores_task = 2
    main(save_results, config_path, random, trials, n_cores_task)

