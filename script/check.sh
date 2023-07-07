#!/bin/bash

############################################################
# USAGE                                                    #
############################################################
usage()
{
usage="
  Usage: $0 {options} <problem size> <kernel>\n
  options:\n
    \t-h,--help : print help\n
    \t-a,-all : check all kernel\n
    \t-v,--verbose : print all make output\n
  problem size: default value = 10
  kernels:\n
    \t${kernel_list[*]}\n"

  echo -e $usage
  exit 1
}

############################################################
# IF ERROR                                                 #
############################################################

check_error()
{
  err=$?
  if [ $err -ne 0 ]; then
    echo -e "gpumm: error in $0\n\t$1 ($err)"
    echo "Script exit."
    exit 1
  fi
}

############################################################
# CHECK REQUIERMENT                                        #
############################################################
check_requierment()
{
  JQ_OK=$(which jq)
  if [ "" = "$JQ_OK" ]; then
    echo -n "jq is needed. Install jq ?"
    while true; do
          read -p " (y/n) " yn
          case $yn in
              [Yy]*) sudo apt-get --yes install jq ; break;; 
              [Nn]*) echo "Aborted" ; exit 1 ;;
          esac
      done
  fi
}

############################################################
# FUNCTION                                                 #
############################################################
check_kernel()
{
  echo -e "Check kernel $1 . . ."
  eval make check -B GPU=$GPU KERNEL=`echo "$1" | tr '[:lower:]' '[:upper:]'` $output
  check_error "make echoué"
  cmd="./check $2 ./output/check_`echo $1 | tr '[:upper:]' '[:lower:]'`.out"
  eval echo "exec command : $cmd" $output
  eval $cmd
  check_error "lancement du programme echoué"
  echo -e "Comparaison outputs base/$1 . . ."
  python3 ./python/files_cmp.py `echo "$1" | tr '[:upper:]' '[:lower:]'`
  check_error "script python echoué"
}

############################################################
# SETUP                                                    #
############################################################

check_requierment
WORKDIR=`realpath $(dirname $0)/..`
cd $WORKDIR

############################################################
# CHECK GPU                                                #
############################################################
GPU_CHECK=$( lshw -C display 2> /dev/null | grep nvidia )
GPU=NVIDIA
if [[ -z "$GPU_CHECK" ]]; then
  GPU_CHECK=$( lshw -C display 2> /dev/null | grep amd )
  GPU=AMD
fi
if [[ -z "$GPU_CHECK" ]]; then
  echo "No GPU found."
  exit 1
fi

############################################################
# CHECK OPTIONS                                            #
############################################################

kernel_list=( $(jq ".$GPU|keys_unsorted[1:][]" json/measure_config.json -r) )
kernel_to_check=()
verbose=0
output="> /dev/null"
all=0

TEMP=$(getopt -o hav \
              -l help,all,verbose \
              -n $(basename $0) -- "$@")

if [ $? != 0 ]; then usage ; fi

eval set -- "$TEMP"

while true ; do
    case "$1" in
        -a|--all) kernel_to_check=${kernel_list[@]} ; shift ;;
        -v|--verbose) verbose=1 ; shift ;;
        --) shift ; break ;;
        -h|--help|*) usage ;;
    esac
done

############################################################
# CHECK ARGS                                               #
############################################################
re='^[0-9]+$'
data_size=100
for i in $@; do 
  if [[ " ${kernel_list[*]} " =~ " ${i} " ]] && [ $all == 0 ]; then
    kernel_to_check+=" $i"
  elif [[ $i =~ $re ]]; then
    data_size="$i"
  fi
done

if [[ ${kernel_to_check[@]} == "" ]]; then
  echo "No kernel to check"
  exit 1
fi
kernel_to_check=`echo "$kernel_to_check" | tr '[:upper:]' '[:lower:]'`

if [ $verbose == 1 ]; then
  output=""
  echo "Verbose mode on"
fi 

############################################################
# START CHECK                                              #
############################################################
echo "Début des vérifications des sorties"
echo -e "Check base kernel . . ."
eval make check KERNEL=BASIS GPU=$GPU -B $output
check_error "make echoué"
eval ./check $data_size "./output/check_basis.out"
check_error "lancement du programme echoué"

for i in $kernel_to_check; do
  kernel=`echo "$i" | tr '[:upper:]' '[:lower:]'`
  if [[ " ${kernel_list[*]} " =~ " ${kernel} " ]]; then
    check_kernel $kernel $data_size
  fi
done

eval make clean $output
echo "Vérifications terminées"
