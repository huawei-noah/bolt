#!/bin/bash
# NOTE: please make sure that you have meet the following requirements
#     1. model in the device directory.
#     2. inference program in the device directory.
#     3. test data in the current host directory.
device_dir=/data/local/test/bolt/tinybert

adb push sequence.txt ${device_dir}/data/sequence.txt
adb shell "cd ${device_dir} && ./tinybert tinybert_f16.bolt data" &> result.txt
