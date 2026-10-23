#! /bin/bash

# Get current timestamp for experiment naming
printf -v now '%(%F_%H-%M-%S)T' -1
exp_name="experiment_$1_$now"
exp_create_str=$(mlflow experiments create -n $exp_name)
echo $exp_create_str
exp_id=$(echo $exp_create_str | awk '{print $NF}')

# Define model architectures to iterate over
models=("ggml-org/gemma-3-4b-it-GGUF"
    "ggml-org/gemma-4-E4B-it-GGUF" "ggml-org/gemma-3-27B-it-GGUF"
    "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF"
    "MaziyarPanahi/Mistral-7B-Instruct-v0.3-GGUF"
        )

dataset="../dataset/dataset_samples/LDJnr_Pure-Dove_splittrain_n1_seedNone_modelgemma3_27b_temperature0.5_flagssfa.jsonl"
#dataset="../dataset/dataset_samples/LDJnr_Pure-Dove_splittrain_n10_seed30_modelgemma3_27b_temperature0.5_flagssfa.jsonl"
#dataset="../dataset/dataset_samples/LDJnr_Pure-Dove_splittrain_n100_seed30_modelgemma3_27b_temperature0.5_flagssfa.jsonl"

backend="llama.cpp"

seed=$(awk -v min=1 -v max=10000 'BEGIN{srand(); print int(min+rand()*(max-min+1))}')
echo "Seed for randomization operations: $seed"

# Loop over all models, temperature and input variants
for m in "${models[@]}"
do
    for t in "0.0" "0.7"
    do
        for f in "input" "simple" "friendly" "academic"
        do
            # Keep trying the mlflow run until it succeeds
            while true
            do
                echo "Running model $m with temperature $t and formulation $f ..."
                timeout -s SIGKILL $(( $1 + 600 )) mlflow run --experiment-name=$exp_name -e main.py -P model=$m -P seconds=$1 -P seed=$seed -P temperature=$t -P dataset=$dataset -P field=$f -P backend=$backend .

                # Check if the mlflow run succeeded (exit status 0)
                if [ $? -eq 0 ]; then
                    echo "Run succeeded for model $m with dataset $d and formulation $f"
                    break  # Exit the loop if succeeded
                else
                    echo "Run failed for model $m, (temp $t, formulation $f) retrying..."
                fi
            done
            # Save experiment data to CSV
        mlflow experiments csv -x $exp_id > "../results/$exp_name.csv"
        done
    done
done
