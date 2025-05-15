#!/bin/bash
#SBATCH --account=def-gdumas85
#SBATCH --nodes 1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --mail-user=davidjlemay@gmail.com
#SBATCH --mail-type=END
#SBATCH --output=%N.%j.out

# https://docs.alliancecan.ca/wiki/Python#Creating_virtual_environments_inside_of_your_jobs
# https://docs.alliancecan.ca/wiki/Huggingface#Training_Large_Language_Models_(LLMs)

# Define cleanup function for error handling
cleanup() {
    echo "Error detected at line $1. Saving any output files before exiting..."
    
    # Create tarball of any results that were generated
    if [ -d "${SLURM_TMPDIR}/results" ]; then
        echo "Archiving results from ${SLURM_TMPDIR}/results"
        tar -czvf ${SLURM_TMPDIR}/results_${HOSTNAME}.${SLURM_JOB_ID}.tar.gz ${SLURM_TMPDIR}/results/*
        rsync -a ${SLURM_TMPDIR}/results_${HOSTNAME}.${SLURM_JOB_ID}.tar.gz ${PWD}/results/
        echo "Outputs saved to ${PWD}/results/results.tar.gz"
    else
        echo "No results directory found at ${SLURM_TMPDIR}/results"
    fi
    
    exit 1
}

# Set trap to catch errors
trap 'cleanup $LINENO' ERR


export HF_HOME=$SCRATCH/hf_models
export HF_HUB_OFFLINE=1
#huggingface-cli download MODEL

echo "loading models from $HF_HOME ..."

module load gcc arrow/15.0.1 opencv/4.11.0 python/3.11


virtualenv --no-download $SLURM_TMPDIR/venv

source $SLURM_TMPDIR/venv/bin/activate

pip install --no-index --upgrade pip
pip install --no-index -r requirements_cc.txt


cp -R /home/dlemay/projects/def-gdumas85/dlemay/ToMBench/data $SLURM_TMPDIR/data
mkdir $SLURM_TMPDIR/results

# Set environment variables for better performance
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16

# Print system information
echo "=== System Information ==="
echo "Running on node: $(hostname)"
echo "CUDA devices: $(nvidia-smi --list-gpus)"
echo "CPU cores: $(nproc)"
echo "Memory: $(free -h | grep Mem | awk '{print $2}')"

# Monitor GPU utilization in the background
(nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total --format=csv -l 60 > $SLURM_TMPDIR/results/gpu_stats_${SLURM_JOB_ID}.csv) &
NVIDIA_SMI_PID=$!

# Run the inference script
python run_vllm.py \
  --model_path "Qwen/Qwen3-1.7B" \
  --input_dir $SLURM_TMPDIR/data \
  --output_dir "./results" \
  --batch_size 32 \
  --prompt_field "STORY" \
  --max_tokens 512 \
  --temperature 0.7 \
  --checkpoint_interval 50 \
  --file_pattern "*.jsonl" \
  --num_rounds 10 \
  --run_id "Qwen/Qwen3-1.7B" \
  --skip_processed \
  --language "en" \
  --cot false

# Kill the background monitoring process
kill $NVIDIA_SMI_PID

echo "Python script completed successfully. Archiving results..."
tar -czvf ${SLURM_TMPDIR}/results_${HOSTNAME}.${SLURM_JOB_ID}.tar.gz ${SLURM_TMPDIR}/results/*
rsync -a ${SLURM_TMPDIR}/results_${HOSTNAME}.${SLURM_JOB_ID}.tar.gz ${PWD}/results/
echo "Job completed successfully!"
