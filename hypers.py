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
    'alpha_explore': 'exploration'
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
    'alpha_explore': 'prior'
}

SUBCONFIG_MAPPING = {
    'alpha_explore': 'alpha',
}

def train_dso(config):
    """Trains DSO and returns dict of reward, expression, and traversal"""

    print("\n== TRAINING SEED {} START ============".format(config["experiment"]["seed"]))

    # For some reason, for the control task, the environment needs to be instantiated
    # before creating the pool. Otherwise, gym.make() hangs during the pool initializer
    ''' # Removed gym error as not using control task
    if config["task"]["task_type"] == "control" and config["training"]["n_cores_batch"] > 1:
        import gym
        import dso.task.control # Registers custom and third-party environments
        gym.make(config["task"]["env"])
    '''

    # Train the model
    model = DeepSymbolicOptimizer(deepcopy(config))
    start = time.time()
    result = model.train()
    result["t"] = time.time() - start
    result.pop("program")

    save_path = model.config_experiment["save_path"]
    summary_path = os.path.join(save_path, "summary.csv")

    print("== TRAINING SEED {} END ==============".format(config["experiment"]["seed"]))

    return result, summary_path


def print_summary(config, runs, messages):
    text = '\n== EXPERIMENT SETUP START ===========\n'
    text += 'Task type            : {}\n'.format(config["task"]["task_type"])
    if config["task"]["task_type"] == "regression":
        text += 'Dataset              : {}\n'.format(config["task"]["dataset"])
    elif config["task"]["task_type"] == "control":
        text += 'Environment          : {}\n'.format(config["task"]["env"])
    text += 'Starting seed        : {}\n'.format(config["experiment"]["seed"])
    text += 'Runs                 : {}\n'.format(runs)
    if len(messages) > 0:
        text += 'Additional context   :\n'
        for message in messages:
            text += "      {}\n".format(message)
    text += '== EXPERIMENT SETUP END ============='
    print(text)


def clean_config(config_template="", runs=1, n_cores_task=1, seed=None, benchmark=None, exp_name=None):
    """Runs DSO in parallel across multiple seeds using multiprocessing."""

    messages = []

    # Load the experiment config
    config_template = config_template if config_template != "" else None
    config = load_config(config_template)

    # Overwrite named benchmark (for tasks that support them)
    task_type = config["task"]["task_type"]
    if benchmark is not None:
        # For regression, --b overwrites config["task"]["dataset"]
        if task_type == "regression":
            config["task"]["dataset"] = benchmark
        # For control, --b overwrites config["task"]["env"]
        elif task_type == "control":
            config["task"]["env"] = benchmark
        else:
            raise ValueError("--b is not supported for task {}.".format(task_type))

    # Update save dir if provided
    if exp_name is not None:
        config["experiment"]["exp_name"] = exp_name

    # Overwrite config seed, if specified
    if seed is not None:
        if config["experiment"]["seed"] is not None:
            messages.append(
                "INFO: Replacing config seed {} with command-line seed {}.".format(
                    config["experiment"]["seed"], seed))
        config["experiment"]["seed"] = seed

    # Save starting seed and run command
    config["experiment"]["starting_seed"] = config["experiment"]["seed"]
    config["experiment"]["cmd"] = " ".join(sys.argv)

    # Set timestamp once to be used by all workers
    timestamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
    config["experiment"]["timestamp"] = timestamp

    # Fix incompatible configurations
    if n_cores_task == -1:
        n_cores_task = multiprocessing.cpu_count()
    if n_cores_task > runs:
        messages.append(
                "INFO: Setting 'n_cores_task' to {} because there are only {} runs.".format(
                    runs, runs))
        n_cores_task = runs
    if config["training"]["verbose"] and n_cores_task > 1:
        messages.append(
                "INFO: Setting 'verbose' to False for parallelized run.")
        config["training"]["verbose"] = False
    if config["training"]["n_cores_batch"] != 1 and n_cores_task > 1:
        messages.append(
                "INFO: Setting 'n_cores_batch' to 1 to avoid nested child processes.")
        config["training"]["n_cores_batch"] = 1
    if config["gp_meld"]["run_gp_meld"] and n_cores_task > 1 and runs > 1:
        messages.append(
                "INFO: Setting 'parallel_eval' to 'False' as we are already parallelizing.")
        config["gp_meld"]["parallel_eval"] = False

    # Start training
    print_summary(config, runs, messages)

    return config, runs, n_cores_task

def run_experiment(config, runs, n_cores_task):
    # Generate configs (with incremented seeds) for each run
    configs = [deepcopy(config) for _ in range(runs)]
    for i, config in enumerate(configs):
        config["experiment"]["seed"] += i

    # Farm out the work
    if n_cores_task > 1:
        pool = multiprocessing.Pool(n_cores_task)
        for i, (result, summary_path) in enumerate(pool.imap_unordered(train_dso, configs)):
            if not safe_update_summary(summary_path, result):
                print("Warning: Could not update summary stats at {}".format(summary_path))
            print("INFO: Completed run {} of {} in {:.0f} s".format(i + 1, runs, result["t"]))
    else:
        for i, config in enumerate(configs):
            result, summary_path = train_dso(config)
            if not safe_update_summary(summary_path, result):
                print("Warning: Could not update summary stats at {}".format(summary_path))
            print("INFO: Completed run {} of {} in {:.0f} s".format(i + 1, runs, result["t"]))

    # Evaluate the log files
    print("\n== POST-PROCESS START =================")
    log = LogEval(config_path=os.path.dirname(summary_path))
    log.analyze_log(
        show_count=config["postprocess"]["show_count"],
        show_hof=config["logging"]["hof"] is not None and config["logging"]["hof"] > 0,
        show_pf=config["logging"]["save_pareto_front"],
        save_plots=config["postprocess"]["save_plots"])
    print("== POST-PROCESS END ===================")
    return summary_path

def grid_search(config, param_dicts):
    summaries = []
    timestamp = None
    print(f"INFO: RUNNING {len(param_dicts)} EXPERIMENTS")
    for i, params in enumerate(param_dicts):
        config_mod = copy.deepcopy(config)

        exp_suffix = ""
        for param in params:
            if isinstance(config_mod[CONFIG_MAPPING[param]][PARAM_MAPPING[param]], str):
                config_mod[CONFIG_MAPPING[param]][PARAM_MAPPING[param]] = params[param]
            elif isinstance(config_mod[CONFIG_MAPPING[param]][PARAM_MAPPING[param]], dict):
                config_mod[CONFIG_MAPPING[param]][PARAM_MAPPING[param]][SUBCONFIG_MAPPING[param]] \
                    = params[param]
            else:
                raise KeyError(f'Error: Please check the format of config dict for parameter {param}.')

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

        print(f"\n=== Running grid search with {params} ===")

        summary_path = run_experiment(config_mod, runs, n_cores_task)

        parameters = json.dumps(params)
        summary = pd.read_csv(summary_path)

        summary["params_json"] = parameters
        summaries.append(summary)
        t = summary["t"]
        print(f"=== FINISHED ITERATION {i} in {float(t): .4f} seconds===")
        print(summary)

    return summaries, timestamp

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

    # Exlore Parameters
    alpha_explore = [0, 0.001, 0.01, 0.1, 0.2, 0.5, 0.7, 1, 5, 10]

    # param_dicts = [
    #     {"lr": lr, "ew": ew, "eg": eg, "clip": clip, "iters": iters, "mb": mb}
    #     for lr, ew, eg, clip, iters, mb in
    #     itertools.product(learning_rates, entropy_weights, entropy_gammas, ppo_clip_ratio, ppo_n_iters, ppo_n_mb)
    # ]

    # For testing
    param_dicts = [
        {"alpha_explore": at}
        for at in alpha_explore
    ]

    start = time.time()
    if random:
        if trials is None:
            raise ValueError("Must provide trials when random is True.")
        param_dicts = np.random.choice(param_dicts, trials, replace=False)
        summaries, timestamp = grid_search(config, param_dicts)
    else:
        summaries, timestamp = grid_search(config, param_dicts)
    end = time.time()
    print(f"Time taken to run search: {end - start: .4f} seconds")

    all_results = pd.concat(summaries, ignore_index=True)
    all_results_sorted = all_results.sort_values(by="t", ascending=True)

    print(all_results_sorted)
    if save_results:
        folder = f'./log/hypers_{timestamp}'
        os.makedirs(folder, exist_ok=True)
        print(f"Saving results to {folder}...")
        all_results_sorted.to_csv(f'{folder}/results.csv', index=False)

if __name__ == "__main__":
    save_results = True
    config_path = '/homes/55/panu/4yp/deep-symbolic-optimization/dso/dso/config/config_regression.json'
    random = False
    trials = 10
    main(save_results, config_path, random, trials)

