adbDeviceNum=($(adb devices | sed 's/\r//' | grep ".device$"))
i=0
length=${#adbDeviceNum[@]}
while [ "$i" -lt "$length" ]; do
    if 
        ((i%2!=0))
    then
        unset adbDeviceNum[i]
    fi
    ((i++))
done
adbDeviceNum=(E5B0119506000260 GCL5T19822000030)
