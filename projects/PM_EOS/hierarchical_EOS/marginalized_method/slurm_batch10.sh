#!/bin/bash
#SBATCH --job-name=run_payne      # Job name
#SBATCH --output=log/slurm-%A_%a.out      # Standard output and error log, %A is job ID, %a is array task ID
#SBATCH --error=log/slurm-%A_%a.err       # Standard error log
#SBATCH --nodes=1                     # Run all tasks on a single node
#SBATCH --ntasks-per-node=1           # Number of tasks per node
#SBATCH --cpus-per-task=2             # Number of CPU cores per task
#SBATCH --mem=4G                      # Memory per node
#SBATCH --time=01:40:00               # Time limit hrs:min:sec
#SBATCH --array=0-9                   # Run a job array with task IDs 0 to 9

# --- Environment Setup ---
## Load Anaconda module (adjust as per your cluster's setup)
#module purge
#module load anaconda3/2023.07.1      # Check your cluster's available module names/versions

# Activate your specific conda environment
#conda activate jaxGPU13

# --- Job Execution ---
# Access the unique task ID using the $SLURM_ARRAY_TASK_ID environment variable
printf -v TASK_ID "%03d" "$SLURM_ARRAY_TASK_ID"

# Run your Python script with the task ID as an argument (or use it to select input files)
# -u causes unbuffered flush to output
python -u ./generate_single_logl_R1d6.py $TASK_ID
