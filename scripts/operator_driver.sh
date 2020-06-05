#!/bin/bash

script_name=$0
script_abs=$(readlink -f "$0")
script_dir=$(dirname $script_abs)

cpu_mask="2"
exe_host_path=""
parameter_file_path=""
excute_on_device=false
use_static_library=false
device=""
device_dir=""
exe_device_path=""


print_help() {
    cat <<EOF
Usage: ${script_name} [OPTION]...
Run operator test.

Mandatory arguments to long options are mandatory for short options too.
  -h, --help                 display this help and exit.
  -e, --exe <PATH>           run specified program.
  -i, --input <PATH>         parameter file PATH.
  -s, --static <false|true>  use the static library(default: false).
  -c, --cpu_mask <mask>      taskset cpu mask(default: 2).
  -d, --device <device_id>   run test on device.
  -p, --path <PATH>          run test on device in specified <PATH>.
EOF
    exit 1;
}

TEMP=`getopt -o c:d:e:i:p:hs: --long cpu_mask:device:exe:input:path:help,static: \
     -n ${script_name} -- "$@"`
if [ $? != 0 ] ; then echo "[ERROR] terminating..." >&2 ; exit 1 ; fi
eval set -- "$TEMP"
while true ; do
    case "$1" in
        -c|--cpu_mask)
            cpu_mask=$2
            echo "[INFO] CPU mask '${cpu_mask}'" ;
            shift 2 ;;
        -d|--device)
            device=$2
            exe_on_device=true
            echo "[INFO] test on device '${device}'" ;
            shift 2 ;;
        -p|--path)
            device_dir=$2
            echo "[INFO] test on device path '${device_dir}'" ;
            shift 2 ;;
        -s|--static)
            use_static_library=$2
            echo "[INFO] use static library: ${use_static_library}" ;
            shift 2;;
        -e|--exe)
            exe_host_path=$2
            echo "[INFO] exe '${exe_host_path}'" ;
            shift 2 ;;
        -i|--input)
            parameter_file_path=$2
            echo "[INFO] parameter \`${parameter_file_path}'" ;
            shift 2 ;;
        -h|--help)
            print_help ;
            shift ;;
        --) shift ;
            break ;;
        *) echo "[ERROR]" ; exit 1 ;;
    esac
done

if [ "${exe_host_path}" == "" ] || [ ! -f ${exe_host_path} ] ; then
    echo "[ERROR] exe '${exe}' doesn't exist";
    exit 1
fi

if [ "${parameter_file_path}" == "" ] || [ ! -f ${parameter_file_path} ] ; then
    echo "[ERROR] parameter '${parameter_file_path}' doesn't exist";
    exit 1
fi

if [ ${exe_on_device}  == true ] ; then
    exe_name=${exe_host_path##*/}
    exe_device_path="${device_dir}/${exe_name}"
    adb -s ${device} push ${exe_host_path} ${exe_device_path} || exit 1
fi

while read params
do
    # filter out the params that starts with '#'
    if [[ ! "$params" =~ ^#.* ]]; then
        params_len=${#params}
        if [[ $params_len -gt 0 ]]; then
            #echo "    parameter: ${params}"
            if [ ${exe_on_device} == true ] ; then
                if [ ${use_static_library} == true ] ; then
                    adb -s ${device} shell "taskset ${cpu_mask} ${exe_device_path} ${params} || echo '[FAILURE]'" &> status.txt
                else
                    adb -s ${device} shell "export LD_LIBRARY_PATH=${device_dir} && taskset ${cpu_mask} ${exe_device_path} ${params} || echo '[FAILURE]'" &> status.txt
                fi
            else
                if [ ${use_static_library} == true ] ; then
                    ${exe_host_path} ${params} || echo '[FAILURE]' &> status.txt
                else
                    export LD_LIBRARY_PATH=${exe_host_path}/../lib:${LD_LIBRARY_PATH} && ${exe_host_path} ${params} || echo '[FAILURE]' &> status.txt
                fi
            fi
            cat status.txt || exit 1
            if [ `grep -c "\[FAILURE\]" status.txt` -ne '0' ] ; then
                exit 1
            fi
            rm status.txt
        fi
    fi
done < ${parameter_file_path}

if [ ${exe_on_device}  == true ] ; then
    adb -s ${device} shell "rm -rf ${exe_device_path}"
fi
