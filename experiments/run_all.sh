#! /bin/bash

# Get current timestamp for experiment naming
printf -v now '%(%F_%H-%M-%S)T' -1
exp_name="experiment_$1_$now"
exp_create_str=$(mlflow experiments create -n $exp_name)
echo $exp_create_str
exp_id=$(echo $exp_create_str | awk '{print $NF}')

# Define model architectures to iterate over # replace 8b with 14b
models=("gemma3:4b" #"gemma3:270m" "gemma3n:e2b" "gemma3:27b"
        "tinyllama:1.1b" # "llama4:scout"
        "mistral:7b") #"mistral-medium-3.5:128b" "magistral:24b")

dataset="../dataset/dataset_samples/LDJnr_Pure-Dove_splittrain_n10_seed30_modelgemma3_27b_temperature0.5_flagssfa.jsonl"

# Loop over all models and GPUs
for m in "${models[@]}"
do
    for t in "0.1" "0.7"
    do
        for f in "input" "simple" "friendly" "academic"
        do
            # Keep trying the mlflow run until it succeeds
            while true
            do
                echo "Running model $m on GPU $g ..."
                timeout $(( $1 * 3 )) mlflow run --experiment-name=$exp_name -e main.py -P model=$m -P seconds=$1 -P temperature=$t -P dataset=$dataset -P field=$f .
            
                # Check if the mlflow run succeeded (exit status 0)
                if [ $? -eq 0 ]; then
                    echo "Run succeeded for model $m on GPU $g with dataset $d and formulation $f"
                    break  # Exit the loop if succeeded
                else
                    echo "Run failed for model $m on GPU $g, retrying..."
                fi
            done
            # Save experiment data to CSV
        mlflow experiments csv -x $exp_id > "../results/$exp_name.csv"
        done
    done
done
