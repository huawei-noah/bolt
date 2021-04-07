for file in *
    do
        if [ "${file##*.}"x = "cl"x ];then
            if [[ "${file}" == "depth2space_ncwhc4_2x2.cl" ]];then
                 echo ./gcl_binary --input=$file --output=${file%.*}.bin --options=\"${copt} \"
                 echo ./gcl_binary --input=$file --output=${file%.*}_nchw.bin --options=\"${copt} -DOUT_NCHW\"
            fi
        fi
    done



