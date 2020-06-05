adbDeviceNum=($(adb devices | grep ".device$"))
i=0
length=${#adbDeviceNum[@]}
while [ "$i" -lt "$length" ];do
    if 
        ((i%2!=0)) 
    then
        unset adbDeviceNum[i]  
    fi
    ((i++))
done
