#set your adb device Num in file adbDeviceNum#
#it support mult devices, if they have been connected to your host#
source sh/adbDeviceNum.sh
source sh/buildKernelLibConfig.sh

#build kernel bin on device                                  #
#if your host can not support these devices you need one time#
#please execute buildKernelBin.sh mult times                 #
#with adbDeviceNum update #
source sh/buildKernelBin.sh

#after build all the kernel bin you need for diffrent device#
#execute sh/packKernelBin.sh to build head file and lib.a#
source sh/packKernelBin.sh

#clean workPath#
rm -rf bin cl name
rm bin2char gcl_binary gcl_device_name   

