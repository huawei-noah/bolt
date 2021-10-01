if [ $# -ne 1 ]
then
    echo "Please set your device Number"
    exit 1
fi
adbDeviceNum=$1
buildPath=../../../../build_android-aarch64/common/gcl/tools/gcl_sample
runPath=/data/local/tmp
testName=gcl_sample

cd ${buildPath}
make -j33
adb -s ${adbDeviceNum} push ${testName} ${runPath}
adb -s ${adbDeviceNum} shell "cd ${runPath} && chmod +x ${testName} && ./${testName}"

