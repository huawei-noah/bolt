for file in *
    do
        if [ "${file##*.}"x = "cl"x ];then
            if [[ "${file}" == "conv_direct_spe_fwhs1.cl" ]];then
                echo ./gcl_binary --input=$file --output=${file%.*}_1.bin --options=\"${copt} -D OC=1\"
                echo ./gcl_binary --input=$file --output=${file%.*}_2.bin --options=\"${copt} -D OC=2\"
                echo ./gcl_binary --input=$file --output=${file%.*}_3.bin --options=\"${copt} -D OC=3\"
                echo ./gcl_binary --input=$file --output=${file%.*}_4.bin --options=\"${copt} -D OC=4\"
                echo ./gcl_binary --input=$file --output=${file%.*}_8.bin --options=\"${copt} -D OC=8\"
                echo ./gcl_binary --input=$file --output=${file%.*}_16.bin --options=\"${copt} -D OC=16\"
                echo ./gcl_binary --input=$file --output=${file%.*}_relu_4.bin --options=\"${copt} -D OC=4 -D USE_RELU\"
                echo ./gcl_binary --input=$file --output=${file%.*}_relu_8.bin --options=\"${copt} -D OC=8 -D USE_RELU\"
                echo ./gcl_binary --input=$file --output=${file%.*}_relu_16.bin --options=\"${copt} -D OC=16 -D USE_RELU\"
            fi
        fi
    done



