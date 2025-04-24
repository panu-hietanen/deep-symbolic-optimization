"""Parallelized, single-point launch script to run DSO on a set of benchmarks."""
import csv
import os
import time
import commentjson as json
import numpy as np
import pandas as pd
import copy
import itertools
from functools import reduce
import re

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

def benchmark(config, benchmarks, runs=1, n_cores_task=1, recovery_files=None):
    summaries = []
    paths = []
    cached = []
    infos = {}
    timestamp = None

    try:
        if recovery_files is not None:
            print('Attempting to recover files.')
            for path in recovery_files:
                summary_path = os.path.join(path, 'summary.csv')
                filepaths = (summary_path, None)

                config_path = os.path.join(path, 'config.json')
                try:
                    with open(config_path, encoding='utf-8') as f:
                        config = json.load(f)
                except Exception as e:
                    raise ValueError(f'Error reading config file {config_path}: {e}')
                    break
                config_mod, runs, n_cores_task, messages = clean_config(config, runs=runs, n_cores_task=n_cores_task)
                experiment = {
                    "config_mod": config_mod,
                    "benchmark": config_mod['task']['dataset'],
                    "n_cores_task": n_cores_task,
                    "runs": runs
                }

                summaries, cached, infos = handle_summary(experiment, summaries, infos,
                                                          cached, filepaths, recovery=True)
    except Exception as e:
        print("WARNING: Couldn't recover files!")
        print(f"Error {type(e).__name__}: "
              f"{e}.")

    n = len(benchmarks)
    print(f"INFO: RUNNING {n} BENCHMARKS {runs} TIMES")
    try:
        for i, benchmark in enumerate(benchmarks):
            print(f"\n=== Dataset {benchmark} ({i}/{n}) ===")
            config_mod = copy.deepcopy(config)

            exp_suffix = benchmark

            config_mod["task"]["dataset"] = benchmark

            config_mod, runs, n_cores_task = clean_config(config_mod, runs=runs, n_cores_task=n_cores_task)
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
            summary_path, output_prefix = run_experiment(config_mod, runs, n_cores_task)
            end = time.time()

            filepaths = (summary_path, output_prefix)
            paths.append(filepaths)
            experiment = {
                "config_mod": config_mod,
                "benchmark": benchmark,
                "n_cores_task": n_cores_task,
                "runs": runs
            }
            summaries, cached, infos = handle_summary(experiment, summaries, infos, cached, filepaths)

            print(f"=== FINISHED BENCHMARK {benchmark} IN {end - start: .4f} SECONDS===")
    except KeyboardInterrupt:
        print("Interrupted by user. Saving...")
    except Exception as e:
        print(f"Error {type(e).__name__}: {e}. Trying to recover...")
        summaries = []
        timestamp = "RECOVERY"
        for path, benchmark in zip(paths[:-1], benchmarks[:-1]):
            experiment = {
                "config_mod": config_mod,
                "benchmark": benchmark,
                "n_cores_task": n_cores_task,
                "runs": runs
            }
            summaries, cached, infos = handle_summary(experiment, summaries, infos, cached, path, recovery=True)

    return summaries, cached, infos, timestamp

def handle_summary(experiment, summaries, infos, cached, filepaths, recovery=False):
    summary_path, output_prefix = filepaths
    summary = pd.read_csv(summary_path)
    if experiment["config_mod"]["logging"]["save_cache"] and not recovery:
        cache_file = output_prefix + "_cache.csv"
        try:
            cache = pd.read_csv(cache_file)
            cached.append(cache)
        except FileNotFoundError:
            print('Warning: Cache file not found.')
    if experiment["config_mod"]["logging"]["save_all_iterations"] and not recovery:
        indexed_output = re.sub(r'_\d{1,2}$', '', output_prefix)
        infos_per_iteration = []
        try:
            for i in range(experiment["runs"]):
                info_file = f'{indexed_output}_{i}_all_info.csv'
                info = pd.read_csv(info_file)
                info_per_iteration = info.groupby('iteration').agg(
                    r_max=('r', 'max'),
                    r_min=('r', 'min'),
                    r_mean=('r', 'mean'),
                ).reset_index()
                infos_per_iteration.append(info_per_iteration)

            if experiment["config_mod"]["logging"]["save_all_iterations_detailed"]:
                detailed_infos_per_iteration = []
                try:
                    for i in range(experiment["runs"]):
                        detailed_info_file = f'{indexed_output}_{i}_all_info_detailed.csv'
                        detailed_info = pd.read_csv(detailed_info_file)
                        detailed_info_per_iteration = detailed_info.groupby('iteration').agg(
                            r_mean_all=('r', 'mean'),
                        ).reset_index()

                        detailed_infos_per_iteration.append(detailed_info_per_iteration)
                    temp = []
                    for (i, i_det) in zip(infos_per_iteration, detailed_infos_per_iteration):
                        info_comb = i.join(i_det.set_index("iteration"), on="iteration")
                        temp.append(info_comb)
                    infos_per_iteration = temp
                except FileNotFoundError:
                    print('Warning: Detailed info file not found.')
            if experiment["runs"] == 1:
                if experiment['benchmark'] in infos:
                    infos[experiment['benchmark']].append(infos_per_iteration[0])
                else:
                    infos[experiment['benchmark']] = [infos_per_iteration[0]]
            else:
                if experiment['benchmark'] in infos:
                    infos[experiment['benchmark']] += infos_per_iteration
                else:
                    infos[experiment['benchmark']] = infos_per_iteration
        except FileNotFoundError:
            print('Warning: Info file not found.')

    summary["dataset"] = experiment['benchmark']
    summary["sync"] = experiment["config_mod"]["training"]["sync"]
    summary["workers"] = experiment["n_cores_task"] if experiment["config_mod"]["training"]["sync"] else 0
    summaries.append(summary)

    return summaries, cached, infos

def postprocess(summaries, cached, infos, timestamp, config, save_results=False):
    try:
        all_results = pd.concat(summaries, keys=range(len(summaries)))
    except ValueError as e:
        print(f"Error when collecting summaries: {e}")
        return
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

    if infos:
        combined_dfs = []
        for benchmark, dfs in infos.items():
            combined_df = handle_df_list(dfs)
            combined_df["dataset"] = benchmark
            combined_dfs.append(combined_df)

        all_info = pd.concat(combined_dfs)
    else:
        all_info = None

    print("== RESULTS ==")
    print(summary_df)
    if save_results:
        folder = f'./log/bench_{timestamp}'
        os.makedirs(folder, exist_ok=True)
        print(f"Saving results to {folder}...")
        all_results_sorted.to_csv(f'{folder}/results.csv', index=False)
        summary_df.to_csv(f'{folder}/summary.csv', index=False)
        for key, value in config.items():
            with open(f"{folder}/config_{key}.csv", 'w') as f:
                w = csv.DictWriter(f, value.keys())
                w.writeheader()
                w.writerow(value)
        if all_caches_sorted is not None:
            all_caches_sorted.to_csv(f'{folder}/cache.csv', index=False)
        if all_info is not None:
            all_info.to_csv(f'{folder}/all_info.csv', index=False)

def handle_df_list(dfs):
    renamed_dfs = []

    for i, df in enumerate(dfs):
        # Select all columns except 'r_max'
        available_cols = [col for col in df.columns if col != "r_max" and col != "iteration"]

        # Always include 'iteration' for merging
        temp = df[["iteration"] + available_cols].copy()

        # Rename all non-iteration columns with run index suffix
        temp = temp.rename(columns={col: f"{col}:{i}" for col in available_cols})

        renamed_dfs.append(temp)

    # Merge all renamed temp dfs on 'iteration'
    merged_metrics = reduce(lambda left, right: pd.merge(left, right, on="iteration", how="outer"), renamed_dfs)

    # r_max aggregation
    r_max_df = pd.concat([df[["iteration", "r_max"]] for df in dfs if "r_max" in df.columns], ignore_index=True)
    r_max_grouped = r_max_df.groupby("iteration", as_index=False).agg(r_max=("r_max", "max"))

    # r_min aggregation
    r_min_df = pd.concat([df[["iteration", "r_min"]] for df in dfs if "r_min" in df.columns], ignore_index=True)
    r_min_grouped = r_min_df.groupby("iteration", as_index=False).agg(r_min=("r_min", "min"))

    # Merge r_max and r_min
    extremes_df = pd.merge(r_max_grouped, r_min_grouped, on="iteration", how="outer")

    # Final merge
    final_df = pd.merge(extremes_df, merged_metrics, on="iteration", how="outer")
    final_df = final_df.sort_values("iteration").reset_index(drop=True)

    return final_df


def main(save_results=False, config_path='', runs=1, n_cores_task=1, recovery_files=None):
    try:
        with open(config_path, encoding='utf-8') as f:
            config = json.load(f)
    except Exception as e:
        raise ValueError(f'Error reading config file {config_path}: {e}')

    # Benchmarks
    benchmarks = [f'Nguyen-{i}' for i in range(1,13)]
    # benchmarks = ['Nguyen-1', 'Nguyen-11']
    # benchmarks = [f'Jin-{i}' for i in range(1,6)]

    start = time.time()
    summaries, cached, infos, timestamp = benchmark(config, benchmarks, runs, n_cores_task, recovery_files)
    end = time.time()
    print(f"Time taken to run search: {end - start: .4f} seconds")

    if save_results:
        config, _, _ = clean_config(config)
    postprocess(summaries, cached, infos, timestamp, config, save_results)

if __name__ == "__main__":
    save_results = True
    config_path = '/homes/55/panu/4yp/deep-symbolic-optimization/dso/dso/config/config_regression.json'
    runs = 1
    n_cores_task = 1
    recovery_files = None
    main(save_results, config_path, runs, n_cores_task, recovery_files)