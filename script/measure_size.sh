#!/bin/bash

check_error()
{
  if [ $? != 0 ]; then
    exit 1
  fi
}

usage="Usage: $0 <problem size> <nb repeat> <KERNEL>"
output="/dev/null"

if [ $# -ne 3 ]; then
  echo $usage
  exit 1
fi

DIR=`dirname $0`
SCRIPT_DIR=$PWD/$DIR
cd $SCRIPT_DIR

echo "Build en cours . . ."
make clean -C ../ > $output
make calibrate KERNEL=$3 -C ../ > $output
echo "Calibration en cours . . ."
eval ../calibrate $1 $2
echo "Transfert des données vers le tableur google sheet . . ."
python3 ../python/fill_gsheet.py
echo "Graphique généré dans 'https://docs.google.com/spreadsheets/d/1yN0hX6dvBykjH2alkL-RB4YYNF7_aJVJYMGTK0_ynug/edit#gid=0'"
