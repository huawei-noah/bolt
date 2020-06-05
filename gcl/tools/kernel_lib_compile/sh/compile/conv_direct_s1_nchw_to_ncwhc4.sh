for file in *
    do
        if [ "${file##*.}"x = "cl"x ];then
            if [[ "${file}" == "conv_direct_s1_nchw_to_ncwhc4.cl" ]];then
                 echo ./gcl_binary --input=$file --output=${file%.*}_18.bin --options=\"${copt}      -D F=1 -D ON=8 -D Fsq=1  -DUSE_HALF\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_36.bin --options=\"${copt}      -D F=3 -D ON=6 -D Fsq=9  -DUSE_HALF\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_54.bin --options=\"${copt}      -D F=5 -D ON=4 -D Fsq=25 -DUSE_HALF\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_relu_18.bin --options=\"${copt} -D F=1 -D ON=8 -D Fsq=1  -DUSE_HALF -DUSE_RELU\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_relu_36.bin --options=\"${copt} -D F=3 -D ON=6 -D Fsq=9  -DUSE_HALF -DUSE_RELU\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_relu_54.bin --options=\"${copt} -D F=5 -D ON=4 -D Fsq=25 -DUSE_HALF -DUSE_RELU\"
            fi
        fi
    done



