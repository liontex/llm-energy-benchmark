import argparse
import os
import sys
import time

import mlflow
import numpy as np
import pandas as pd
from lamarr_energy_tracker import EnergyTracker, GroundTruthTracker
from util import get_gpu_name, get_processor_name, print_colored_block, infos_from_dataset_path


def read_queries(dataset_path, random=True, seed=None):
    conversations = pd.read_json(
        path_or_buf=os.path.join(os.path.dirname(__file__), dataset_path),
        lines=True,
    )
    if random:
        conversations = conversations.sample(frac=1, random_state=seed)
    return conversations


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Inference benchmarking with ollama LLMs"
    )
    # data and model input
    parser.add_argument("--model", default="gemma3:1b")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--nogpu", type=int, default=0)
    parser.add_argument(
        "--seconds",
        type=int,
        default=0,
        help="number of seconds to profile model on a subset of the data -- 0 process complete",
    )
    parser.add_argument("-d", "--dataset", help="path to the dataset")
    parser.add_argument("-f", "--field", default="input")
    parser.add_argument("-s", "--seed", type=int, help="Seed for shuffling dataset before running the experiment")

    args = parser.parse_args()
    mlflow.log_dict(args.__dict__, "config.json")
    tracker = EnergyTracker()
    ground_tracker = GroundTruthTracker()

    # log important params
    params = {
        "model": args.model,
        "temperature": args.temperature,
        "task": "Inference",
        "architecture": get_processor_name() if args.nogpu else get_gpu_name(),
    }
    for key, val in params.items():
        mlflow.log_param(key, val)

    dataset_infos = infos_from_dataset_path(args.dataset)
    for key, val in dataset_infos.items():
        mlflow.log_param(key, val)

    # if required, disable gpu
    if args.nogpu:
        os.environ["OLLAMA_NO_GPU"] = "1"

    # load data
    queries = read_queries(args.dataset, seed=args.seed)
    queries[args.field]

    # delete old models (for freeing disk space), load new model and its meta info, init inference (often has some temporal overhead)
    import ollama

    if len(ollama.list().models) > 0:
        if len(ollama.list().models) > 1 or ollama.list().models[0].model != args.model:
            for mod in ollama.list().models:
                ollama.delete(mod.model)
    ollama.pull(args.model)
    resp = ollama.chat(
        model=args.model,
        messages=[{"role": "user", "content": f"Can you answer questions?"}],
    )
    mlflow.log_param("file_size", ollama.list().models[0].size)
    mlflow.log_param("parameters", ollama.list().models[0].details.parameter_size)

    # run evaluations but watch for time limit
    times, n_samples, tokens = [], 0, {"in": [], "out": []}

    # evaluate queries
    ground_tracker.start()
    tracker.start()
    print_colored_block(
        f"STARTING ENERGY PROFILING FOR   {args.model.upper()}   temperature {args.temperature} on   {'CPU' if args.nogpu else 'GPU'}   with dataset {dataset_infos["dataset_name"]} in {args.field} formulation"
    )
    # run inference
    for index, query in queries.iterrows():
        t0 = time.time()
        resp = ollama.chat(
            model=args.model,
            messages=[{"role": "user", "content": query["input"]}],
            options={"temperature": args.temperature},
        )
        try:
            tokens["in"].append(resp["prompt_eval_count"])
            tokens["out"].append(resp["eval_count"])
        except:
            pass
        times.append(time.time() - t0)
        n_samples += 1
        remaining = (
            args.seconds - sum(times)
            if args.seconds and len(times) < 5
            else args.seconds - (sum(times) + np.average(times))
        )
        print(
            f"\rProcessed queries: {n_samples} | Remaining time: {remaining:.1f}s",
            end="",
            flush=True,
        )
        if args.seconds and remaining < 0:
            break
    print_colored_block(
        f"STOPPING ENERGY PROFILING FOR   {args.model.upper()}  temperature {args.temperature} on   {'CPU' if args.nogpu else 'GPU'}   with dataset {dataset_infos["dataset_name"]} in {args.field} formulation",
        ok=False,
    )
    tracker.stop()
    ground_results = ground_tracker.stop()

    # add average amount of tokens if there were any errors:
    tokens["in"] += [np.mean(tokens["in"])] * (n_samples - len(tokens["in"]))
    tokens["out"] += [np.mean(tokens["out"])] * (n_samples - len(tokens["in"]))

    # assess resource consumption
    emissions = "~/.let/emissions.csv"
    emission_data = pd.read_csv("~/.let/emissions.csv").to_dict()
    results = {
        "n_tokens_in": sum(tokens["in"]),
        "n_tokens_out": sum(tokens["out"]),
        "running_time_total": emission_data["duration"][0],
        "running_time": emission_data["duration"][0] / n_samples,
        "power_draw_total": emission_data["energy_consumed"][0] * 3.6e6,
        "power_draw": emission_data["energy_consumed"][0] * 3.6e6 / n_samples,
        "ground_truth_power_draw_total": ground_results["energy_consumed"] * 3.6e6,
        "ground_truth_power_draw": ground_results["energy_consumed"]
        * 3.6e6
        / n_samples,
    }

    # log results & cleanup
    mlflow.log_param("n_samples", n_samples)
    for key, val in results.items():
        mlflow.log_metric(key, val)
    for f in [emissions]:
        if os.path.isfile(f):
            mlflow.log_artifact(f)
            os.remove(f)
    mlflow.end_run()
    print(results)
    print("n_samples", n_samples)
    sys.exit(0)
