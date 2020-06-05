for file in *
    do
        if [ "${file##*.}"x = "cl"x ];then
            if [[ "${file}" == "conv_wino_trans_outbuf.cl" ]];then
                 echo ./gcl_binary --input=$file --output=${file%.*}.bin --options=\"${copt} \"
                 echo ./gcl_binary --input=$file --output=${file%.*}_relu.bin --options=\"${copt} -D USE_RELU\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_align.bin --options=\"${copt} -D ALIGN\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_relu_align.bin --options=\"${copt} -D ALIGN -D USE_RELU\"
            fi
        fi
    done



