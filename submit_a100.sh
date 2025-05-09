#!/bin/bash
#SBATCH --account=def-gdumas85
#SBATCH --nodes 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100_1g.5gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=6:00:00
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

module load gcc arrow/15.0.1 python/3.11


virtualenv --no-download $SLURM_TMPDIR/venv

source $SLURM_TMPDIR/venv/bin/activate

pip install --no-index --upgrade pip
pip install --no-index -r requirements_cc.txt


#export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512,expandable_segments:True
# Don't restrict to GPU 0 - use both GPUs
# export CUDA_VISIBLE_DEVICES=0,1  # Let the code handle this instead

#export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

cp -R /home/dlemay/projects/def-gdumas85/dlemay/ToMBench/data $SLURM_TMPDIR/data
mkdir $SLURM_TMPDIR/results

python run_huggingface.py \
    --task "" \
    --model_name "Qwen/Qwen3-0.6B" \
    --language "en" \
    --cot False \
    --try_times 5

echo "Python script completed successfully. Archiving results..."
tar -czvf ${SLURM_TMPDIR}/results_${HOSTNAME}.${SLURM_JOB_ID}.tar.gz ${SLURM_TMPDIR}/results/*
rsync -a ${SLURM_TMPDIR}/results_${HOSTNAME}.${SLURM_JOB_ID}.tar.gz ${PWD}/results/
echo "Job completed successfully!"
