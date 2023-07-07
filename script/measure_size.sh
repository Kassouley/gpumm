#!/bin/bash

check_kernel()
{
  echo -e "Check kernel $1 . . ."
  make check -B KERNEL=`echo "$1" | tr '[:lower:]' '[:upper:]'`  > $output
  have_error "make echoué"
  cmd="./check $2 ./output/check_`echo $1 | tr '[:upper:]' '[:lower:]'`.out"
  echo "exec command : $cmd" > $output
  eval $cmd
  have_error "lancement du programme echoué"
  echo -e "Comparaison outputs base/$1 . . ."
  python3 ./python/files_cmp.py `echo "$1" | tr '[:upper:]' '[:lower:]'`
  have_error "script python echoué"
}

have_error()
{
  if [ $? -ne 0 ]; then
    echo "Erreur : $1"
    echo "Script exit"
    exit $?
  fi
}

WORKDIR=`realpath $(dirname $0)/..`
cd $WORKDIR

kernel_list=( "cpu_omp cblas gpu_omp gpu_omp_wo_dt hip rocblas cuda" )
usage="
Usage: $0 {options} <problem size> <kernel>\n
options:\n
  \t-h,--help : print help\n
  \t-a,-all : check all kernel\n
  \t-v,--verbose : print all make output\n
problem size: default value = 10
kernels:\n
  \t${kernel_list[*]}\n"
  
re='^[0-9]+$'
ARG=()
data_size=10
output="/dev/null"
all=0
while [ 1 ]; do
  if [ "$1" = "" ]; then
      break
  elif [ "$1" = "--help" -o "$1" = "-h" ]; then
      echo -e $usage
      exit 1
  elif [[ $1 =~ $re ]]; then
      data_size="$1"
      shift 1
  elif [ "$1" = "--verbose" -o "$1" = "-v" ]; then
      output="/dev/stdout"
      shift 1
  elif [ "$1" = "--all" -o "$1" = "-a" ]; then
      all=1
      shift 1
  else
      ARG+=" $1"
      shift 1
  fi
done

if [ $all == 1 ]; then
  kernel_to_check=${kernel_list[@]}
else
  kernel_to_check=$ARG
fi

if [[ ${kernel_to_check[@]} == "" ]]; then
  echo "No kernel to check"
  echo -e $usage
  exit 1
fi

echo "Début des vérifications des sorties"
echo -e "Check base kernel . . ."
make check KERNEL=BASIS -B > $output
have_error "make echoué"
eval ./check $data_size "./output/check_basis.out"
have_error "lancement du programme echoué"

for i in $kernel_to_check; do
  kernel=`echo "$i" | tr '[:upper:]' '[:lower:]'`
  if [[ " ${kernel_list[*]} " =~ " ${kernel} " ]]; then
    check_kernel $kernel $data_size
  fi
done

make clean > $output
echo "Vérifications terminées"
