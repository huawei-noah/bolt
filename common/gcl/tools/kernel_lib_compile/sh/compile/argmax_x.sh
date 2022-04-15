for file in *
    do
        if [ "${file##*.}"x = "cl"x ];then
            if [[ "${file}" == "argmax_x.cl" ]];then
                 echo ./gcl_binary --input=$file --output=${file%.*}.bin       --options=\"${copt}\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_index.bin --options=\"${copt} -DUSE_INDEX\"
            fi
        fi
    done



