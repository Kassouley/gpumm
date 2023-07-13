#!/bin/bash

############################################################
# USAGE                                                    #
############################################################
usage()
{
  usage="
  \nUsage: $0 {options} [nb step] <kernels>\n\n
  options:\n
    \t-h,--help : print help\n
    \t-a,--all : calibrate all kernel\n
    \t-M,--msize={XX,YY,...} : problem size to calibrate seperate with comma
    \t-p,--plot={plot_file} : create a png plot with the results in png file in argument (default: ./graphs/graph_DATE.png)\n
    \t-v,--verbose : print all make output\n
    \t-f,--force : do not ask for starting the measure\n
  nb step :\n\tdefault value = 100\n
  problem size :\n\tdefault value = 100\n
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
calibrate_kernel()
{
  kernel=`echo "$1" | tr '[:lower:]' '[:upper:]'`
  echo -e "Build kernel $1 . . ."
  eval make calibrate -B KERNEL=$kernel CLOCK=RDTSC GPU=$GPU $output
  check_error "build failed"
  echo -e "Calibrate kernel $1 (problem size: $3, nb step: $2) . . ."
  cmd="$WORKDIR/calibrate $3 $2 $output_file"
  eval echo "exec command : $cmd" $output
  eval $cmd
  check_error "run calibrate failed"
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

kernel_list=( $(jq ".$GPU|keys_unsorted[]" json/measure_config.json -r) )
kernel_to_calibrate=()
data_size=100
verbose=0
output="> /dev/null"
force=0
all=0
plot=0
clock="RDTSC"
clock_label="RDTSC-cycles"
plot_file="$WORKDIR/graphs/calibrate_graph_$(date +%F-%T).png"

TEMP=$(getopt -o hfavM:p:: \
              -l help,force,all,verbose,msize:,plot:: \
              -n $(basename $0) -- "$@")

eval set -- "$TEMP"

if [ $? != 0 ]; then usage ; fi

while true ; do
    case "$1" in
        -a|--all) kernel_to_calibrate=${kernel_list[@]} ; shift ;;
        -f|--force) force=1 ; shift ;;
        -v|--verbose) verbose=1 ; shift ;;
        -p|--plot) 
            case "$2" in
                "") plot=1; shift 2 ;;
                *)  plot=1; plot_file="$2" ; shift 2 ;;
            esac ;;
        -M|--msize) data_size=( $(echo "$2" | sed "s/,/ /g") ); shift 2 ;;
        --) shift ; break ;;
        -h|--help|*) echo "No option $1."; usage ;;
    esac
done

############################################################
# CHECK ARGS                                               #
############################################################
re='^[0-9]+$'
nb_step=100
for i in $@; do 
  if [[ " ${kernel_list[*]} " =~ " ${i} " ]] && [ $all == 0 ]; then
    kernel_to_calibrate+=" $i"
  elif [[ $i =~ $re ]]; then
    nb_step="$i"
  fi
done

if [[ ${kernel_to_calibrate[@]} == "" ]]; then
  echo "No kernel to calibrate."
  exit 1
fi
kernel_to_calibrate=`echo "$kernel_to_calibrate" | tr '[:upper:]' '[:lower:]'`

############################################################
# SUMMARY OF THE RUN                                       #
############################################################
echo -n "Summary calibrate on $GPU GPU"
if [ $verbose == 1 ]; then
  output=""
  echo -n " (with verbose mode)"
fi 
echo -e "\nKernel to calibrate :$kernel_to_calibrate"
if [ $plot == 1 ]; then
  echo "Plot will be generated in '$plot_file'"
fi 
duration=$((${#kernel_to_calibrate[@]} * 2 * ${#data_size[@]}));
echo "Approximately $duration minutes long"

if [ $force == 0 ]; then
  echo -n "Starting ?"
  while true; do
          read -p " (y/n) " yn
        case $yn in
            [Yy]*) break ;;
            [Nn]*) echo "Aborted" ; exit 1 ;;
        esac
    done
fi

############################################################
# START CALIBRATION                                        #
############################################################
output_file="$WORKDIR/output/calibration_tmp.out"
if [[ -f $output_file ]]; then
  rm $output_file
fi
touch $output_file

echo "Calibration in progress . . ."
for j in ${data_size[@]}; do
  if [[ $j =~ $re ]]; then
    for i in $kernel_to_calibrate; do
      if [[ " ${kernel_list[*]} " =~ " ${i} " ]]; then
        calibrate_kernel $i $nb_step $j
      fi
    done
  fi
  if [ $plot == 1 ]; then
    echo "Génération du graphique . . ."
    python3 ./python/graph-gen-warmup.py $j $output_file $plot_file $clock_label
    echo "Graphique créé dans le répetoire $WORKDIR/graph/"
  fi
done
echo "Calibration finished"

eval make clean $output




