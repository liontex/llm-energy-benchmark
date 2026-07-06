import argparse
import os
import sys
import time

import mlflow
import numpy as np
import pandas as pd
from lamarr_energy_tracker import EnergyTracker, GroundTruthTracker
from lamarr_energy_tracker.tracker import delete_results
from util import get_gpu_name, get_processor_name, print_colored_block, infos_from_dataset_path
import ollama
from llama_cpp import Llama
from huggingface_hub import scan_cache_dir


def read_queries(dataset_path, random=True, seed=None):
    conversations = pd.read_json(
        path_or_buf=os.path.join(os.path.dirname(__file__), dataset_path),
        lines=True,
    )
    if random:
        conversations = conversations.sample(frac=1, random_state=seed)
    return conversations

def load_model(args):
    if args.backend == "ollama":
        # Cleanup of old model before pulling new one
        if len(ollama.list().models) > 0:
            if len(ollama.list().models) > 1 or ollama.list().models[0].model != args.model:
                for mod in ollama.list().models:
                    ollama.delete(mod.model)
        
        ollama.pull(args.model)
        return ollama.Client(host='http://localhost:11434', timeout=args.timeout)
    elif args.backend == "llama.cpp":
        return Llama.from_pretrained(args.model, "*Q4_K_M.gguf", verbose=False, n_ctx=0, n_gpu_layers=-1, seed=args.seed)
    else:
        raise ValueError(f"Invalid Backend choice: {args.backend}, only ollama and llama.cpp supported")

def chat(client, message, args):
    messages = [{"role": "user", "content": message}]

    if args.backend == "ollama":
        return client.chat(
            model=args.model,
            messages=messages,
            options={"temperature": args.temperature, "seed": args.seed},
            keep_alive=f"{args.timeout}s"
        )
    elif args.backend == "llama.cpp":
        return client.create_chat_completion(messages=messages, temperature=args.temperature, seed=args.seed)
    else:
        raise ValueError(f"Invalid Backend choice: {args.backend}, only ollama and llama.cpp supported")

def get_token(response, args):
    if args.backend == "ollama":
        token_in = response["prompt_eval_count"]
        token_out = response["eval_count"]
    elif args.backend == "llama.cpp":
        token_in = response["usage"]["prompt_tokens"]
        token_out = response["usage"]["completion_tokens"]
    else:
        raise ValueError(f"Invalid Backend choice: {args.backend}, only ollama and llama.cpp supported")

    return (token_in, token_out)


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
    parser.add_argument("-b", "--backend", default="ollama")
    parser.add_argument("-t", "--timeout", type=int, help="Timeout per prompt (in seconds), default is 10m)", default=600)
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
    print(f"Pull model {args.model} by backend {args.backend}...")
    client = load_model(args)
    print(f"Model successfully pulled. Testing Inference...")
    resp = chat(client, "Can you answer questions?", args)
    print(f"Inference successfully tested.")
    mlflow.log_param("file_size", ollama.list().models[0].size)
    mlflow.log_param("parameters", ollama.list().models[0].details.parameter_size)

    print("Remove old benchmarking results from results folder.")
    old_emissions = os.path.expanduser("~/.let")
    if os.path.isfile(old_emissions):
        delete_results(old_emissions)

    # run evaluations but watch for time limit
    times, n_samples, tokens = [], 0, {"in": [], "out": []}
    failed_prompts, failed_time = 0, 0
    keep_alive = args.seconds if args.seconds or args.seconds <= 0 else 600

    print("Start Energy Tracking...")
    # evaluate queries
    ground_tracker.start()
    tracker.start()
    print_colored_block(
        f"STARTING ENERGY PROFILING FOR   {args.model.upper()}   temperature {args.temperature} on   {'CPU' if args.nogpu else 'GPU'}   with dataset {dataset_infos["dataset_name"]} in {args.field} formulation"
    )
    print(
            f"\rProcessed queries: {n_samples} | Remaining time: {args.seconds:.1f}s",
            end="",
            flush=True,
        )
    # run inference
    for index, query in queries.iterrows():
        t0 = time.time()
        try:
            response = chat(client, query[args.field], args)
            t1 = time.time()
        except:
            t1 = time.time()
            failed_prompts += 1
            failed_time += t1 - t0

        try:
            token_in, token_out = get_token(response, args)
            tokens["in"].append(token_in)
            tokens["out"].append(token_out)
        except:
            pass
        n_samples += 1
        times.append(t1 - t0)
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
    code_carbon_results = tracker.stop()
    ground_results = ground_tracker.stop()

    # add average amount of tokens if there were any errors:
    tokens["in"] += [np.mean(tokens["in"])] * (n_samples - len(tokens["in"]))
    tokens["out"] += [np.mean(tokens["out"])] * (n_samples - len(tokens["in"]))

    print(code_carbon_results)

    # assess resource consumption
    results = {
        "n_tokens_in": sum(tokens["in"]),
        "n_tokens_out": sum(tokens["out"]),
        "n_failed_prompts": failed_prompts,
        "failed_time": failed_time,
        "running_time_total": code_carbon_results["duration"],
        "running_time": code_carbon_results["duration"] / n_samples,
        "power_draw_total": code_carbon_results["energy_consumed"] * 3.6e6,
        "power_draw": code_carbon_results["energy_consumed"] * 3.6e6 / n_samples,
        "ground_truth_power_draw_total": ground_results["energy_consumed"] * 3.6e6,
        "ground_truth_power_draw": ground_results["energy_consumed"]
        * 3.6e6
        / n_samples,
    }

    # log results & cleanup
    mlflow.log_param("n_samples", n_samples)
    for key, val in results.items():
        mlflow.log_metric(key, val)

    emissions = os.path.expanduser("~/.let/emissions.csv")
    if os.path.isfile(emissions):
        print("log artifact...")
        mlflow.log_artifact(emissions)

    mlflow.end_run()
    print(results)
    print("n_samples", n_samples)
    sys.exit(0)
