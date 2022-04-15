#!/bin/bash
# NOTE: please make sure that you have meet the following requirements
#     1. model in the device directory.
#     2. inference program in the device directory.
#     3. test data in the current host directory.
device_dir=/data/local/test/bolt/tinybert

adb shell mkdir ${device_dir}/data
adb shell mkdir ${device_dir}/data/input
adb shell mkdir ${device_dir}/data/result
adb push sequence.seq ${device_dir}/data/input/0.seq
adb shell "cd ${device_dir} && ./tinybert -m tinybert_f16.bolt -i data -a CPU_AFFINITY_HIGH_PERFORMANCE" &> result.txt
