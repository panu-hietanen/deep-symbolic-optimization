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

def grid_search(config, param_dicts):
    summaries = []
    paths = []
    cached = []
    timestamp = None
    n = len(param_dicts)
    print(f"INFO: RUNNING {n} EXPERIMENTS")
    try:
        for i, params in enumerate(param_dicts):
            config_mod = copy.deepcopy(config)

            exp_suffix = ""
            for param in params:
                config_mod[CONFIG_MAPPING[param]][PARAM_MAPPING[param]] = params[param]
                exp_suffix += f"{param}-{params[param]}_"
                if param in ['clip', 'iters', 'mb']:
                    config_mod['policy_optimizer']['policy_optimizer_type'] = 'ppo'
            exp_suffix = exp_suffix[:-1]

            config_mod, runs, n_cores_task = clean_config(config_mod)
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

            print(f"\n=== Running grid search with {params} ({i}/{n}) ===")

            summary_path, output_prefix = run_experiment(config_mod, runs, n_cores_task)

            filepaths = (summary_path, output_prefix)
            paths.append(filepaths)
            summaries, cached, t = handle_summary(config_mod, params, summaries, cached, filepaths)

            print(f"=== FINISHED ITERATION {i} in {float(t): .4f} seconds===")
    except KeyboardInterrupt:
        print("Interrupted by user. Saving...")
    except Exception as e:
        print(f"Error {type(e).__name__}: {e}. Trying to recover...")
        summaries = []
        timestamp = "RECOVERY"
        for path, params in zip(paths[:-1], param_dicts[:-1]):
            summaries, cached, _ = handle_summary(config, params, summaries, cached, path, recovery=True)

    return summaries, cached, timestamp

def handle_summary(config, params, summaries, cached, filepaths, recovery=False):
    summary_path, output_prefix = filepaths
    summary = pd.read_csv(summary_path)
    if config["logging"]["save_cache"]:
        cache_file = output_prefix + "_cache.csv"
        try:
            cache = pd.read_csv(cache_file)
            cached.append(cache)
        except FileNotFoundError:
            print('Warning: Cache file not found.')

    parameters = json.dumps(params)
    summary["params_json"] = parameters
    summary["dataset"] = config["task"]["dataset"]
    summaries.append(summary)
    t = summary["t"]

    return summaries, cached, t

def postprocess(summaries, cached, timestamp, save_results=False):
    all_results = pd.concat(summaries, keys=range(len(summaries)))
    all_results.index = all_results.index.droplevel(1)
    all_results_sorted = all_results.sort_values(by="t", ascending=True)

    if cached:
        all_caches = pd.concat(cached)
        all_caches_sorted = all_caches.sort_values(by="r", ascending=False)
    else:
        all_caches_sorted = None

    print(all_results_sorted)
    if save_results:
        folder = f'./log/hypers_{timestamp}'
        os.makedirs(folder, exist_ok=True)
        print(f"Saving results to {folder}...")
        all_results_sorted.to_csv(f'{folder}/results.csv', index=True)
        if all_caches_sorted is not None:
            all_caches_sorted.to_csv(f'{folder}/cache.csv', index=False)

def main(save_results=False, config_path='', random=False, trials=None):
    try:
        with open(config_path, encoding='utf-8') as f:
            config = json.load(f)
    except Exception as e:
        raise ValueError(f'Error reading config file {config_path}: {e}')

    # Training Parameters
    batch_sizes = [500, 1000, 5000]
    epsilons = [0.01, 0.05, 0.1]

    # Vanilla PG Parameters
    learning_rates = [5e-5, 1e-4, 5e-4]
    entropy_weights = [0.01, 0.03, 0.1]
    entropy_gammas = [0.5, 0.75, 0.99]

    # PPO Parameters
    ppo_clip_ratio  = [0.1, 0.2, 0.3]
    ppo_n_iters = [5, 10, 15]
    ppo_n_mb = [1, 4, 8]

    param_dicts = [
        {"lr": lr, "ew": ew, "eg": eg, "clip": clip, "iters": iters, "mb": mb}
        for lr, ew, eg, clip, iters, mb in
        itertools.product(learning_rates, entropy_weights, entropy_gammas, ppo_clip_ratio, ppo_n_iters, ppo_n_mb)
    ]

    # For testing
    param_dicts = [
        {"lr": lr}
        for lr in learning_rates
    ]

    start = time.time()
    if random:
        if trials is None:
            raise ValueError("Must provide trials when random is True.")
        param_dicts = np.random.choice(param_dicts, trials, replace=False)
        summaries, cached, timestamp = grid_search(config, param_dicts)
    else:
        summaries, cached, timestamp = grid_search(config, param_dicts)
    end = time.time()
    print(f"Time taken to run search: {end - start: .4f} seconds")

    postprocess(summaries, cached, timestamp, save_results)

if __name__ == "__main__":
    save_results = True
    config_path = '/homes/55/panu/4yp/deep-symbolic-optimization/dso/dso/config/config_regression.json'
    random = False
    trials = 10
    main(save_results, config_path, random, trials)

