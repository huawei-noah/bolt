#build kernel bin on device#
#if devices name are the same, the build will only execute once#
index=1
for dNum in "${adbDeviceNum[@]}"; do
    adb -s ${dNum} shell "rm -rf ${kernelBuildPath}"
    adb -s ${dNum} shell "mkdir  ${kernelBuildPath}"
    adb -s ${dNum} push gcl_device_name ${kernelBuildPath}
    adb -s ${dNum} shell "cd ${kernelBuildPath} && chmod +x gcl_device_name && ./gcl_device_name"
    adb -s ${dNum} shell "cd ${kernelBuildPath} && cat ${deviceNameFile} >> ${dNum}.dn"
    adb -s ${dNum} pull ${kernelBuildPath}/${dNum}.dn ${namePath}
    dname=$(awk '{print $1}' ${namePath}/${dNum}.dn)
    deviceNamesAll[$index]="${dname}"
    dnameS=0
    for((j=1;j<index;j++)); do
        dnameJ=${deviceNamesAll[$j]};
        if [ "$dnameJ" = "$dname" ];then
            dnameS=1
        fi
    done
    if [ $dnameS -eq 0 ]; then
        rm -rf ${binPath}/${dname}
        mkdir  ${binPath}/${dname}
        adb -s ${dNum} shell "cd ${kernelBuildPath} && mkdir sh"
        adb -s ${dNum} push gcl_binary                ${kernelBuildPath}
        adb -s ${dNum} push ${clPath}                 ${kernelBuildPath}
        adb -s ${dNum} push ${compileConfigPath}      ${kernelBuildPath}
        adb -s ${dNum} push ${shPath}/sh.config       ${kernelBuildPath}
        adb -s ${dNum} shell "cd ${kernelBuildPath} && chmod +x gcl_binary"
        adb -s ${dNum} shell "cd ${kernelBuildPath} && cp *.sh ./sh"
        for compileConfig in $compileConfigFiles 
        do
        adb -s ${dNum} shell "cd ${kernelBuildPath} && source ./sh.config && chmod +x ${compileConfig} && ./${compileConfig} > tmp.sh && chmod +x tmp.sh && ./tmp.sh"
        done
        adb -s ${dNum} shell "cd ${kernelBuildPath} && mkdir bin"
        adb -s ${dNum} shell "cd ${kernelBuildPath} && cp *.bin ${kernelBuildPath}/bin"
        adb -s ${dNum} pull ${kernelBuildPath}/bin/ ${binPath}/${dname}
        adb -s ${dNum} shell "rm -rf ${kernelBuildPath}"
        echo ${dname} >> ${dNameFile}
    fi
    index=`expr $index + 1`
done
