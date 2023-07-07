#!/bin/bash
#SBATCH -N 2
#SBATCH -n 160
#SBATCH --gres=gpu:2 
#SBATCH -p small
#SBATCH --ntasks-per-node=80
#SBATCH --time=00:10:00

# workdir=$PWD/slurm/${SLURM_JOBID}
# mkdir ${workdir}

driver=$1
shift 1

echo "Chargement des modules"
module purge
module load nvidia
module load nvhpc/23.1
module load python-tools/3.10.9
module load openblas/gnu/0.3.21
echo "Chargement des modules terminés"

# nsys profile -o ${workdir}/${SLURM_JOBID}-rep ./script/$driver.sh $@
eval ./script/$driver.sh $@
# nsys stats ${workdir}/${SLURM_JOBID}-rep.nsys-rep

sleep 10
jobinfo "${SLURM_JOBID}"



