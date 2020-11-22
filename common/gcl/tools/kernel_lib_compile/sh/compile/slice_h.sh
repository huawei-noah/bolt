for file in *
    do
        if [ "${file##*.}"x = "cl"x ];then
            if [[ "${file}" == "slice_h.cl" ]];then
                 echo ./gcl_binary --input=$file --output=${file%.*}_2.bin --options=\"${copt} -D N=2 \"
            fi
        fi
    done



