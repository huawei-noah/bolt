#!/bin/bash

script_name=$0
script_abs=$(readlink -f "$0")
script_dir=$(dirname $script_abs)

cp ${script_dir}/CPPLINT.cfg $1
cd $1
cpplint --recursive --extensions=cpp,h,hpp,cl .
rm CPPLINT.cfg
echo " "
