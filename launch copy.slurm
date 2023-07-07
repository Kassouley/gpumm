#!/usr/bin/bash
#SBATCH -N 1
#SBATCH -n 16
#SBATCH --ntasks-per-node=16
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --mem=128G
#SBATCH -p big
#nvprofiling


module load amgx/nvidia/2.4.0-nvhpc231-system
module load dcgm/3.1.3-1
module list

workdir=$PWD/${SLURM_JOBID}
mkdir ${workdir}

cd ${workdir}
# \cp -f ../{bul3.*,jcl,PCG.json} .
# cp ../nsys.sh .
cp $0 .
maxit=4


export CUDA_VISIBLE_DEVICES=0,1
export TEST_SYSTEM_PARAMS=1
export UCX_MEMTYPE_CACHE=n

export UCX_IB_GPU_DIRECT_RDMA=no
echo "UCX_IB_GPU_DIRECT_RDMA=${UCX_IB_GPU_DIRECT_RDMA}"
echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"


#DCGM
nv-hostengine --pid nvhostengine.pid --log-filename nv-hostengine.log
#PCIe
dcgmi dmon -e 1009,1010 -c 200 > compteur.${SLURM_JOB_ID} &

clush -w ${SLURM_NODELIST} nvidia-cuda-mps-control -d

BIN=../measure
ldd ${BIN}

nodeset -e ${SLURM_JOBNODELIST} | tr ' ' '\n' > hostfile${SLURM_JOBID}
nsys profile -t mpi,ucx,openmp,openacc,oshmem,cuda,opengl,nvtx,osrt -o report $BIN 500 1 1

nsys stats report.nsys-rep

# rm report*

clush -w ${SLURM_NODELIST}  echo quit | nvidia-cuda-mps-control
kill -9 $(cat nvhostengine.pid)




