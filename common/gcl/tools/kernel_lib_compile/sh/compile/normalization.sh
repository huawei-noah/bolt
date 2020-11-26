for file in *
    do
        if [ "${file##*.}"x = "cl"x ];then
            if [[ "${file}" == "normalization.cl" ]];then
                 echo ./gcl_binary --input=$file --output=${file%.*}_c1.bin --options=\"${copt} -D USE_C1 \"
                 echo ./gcl_binary --input=$file --output=${file%.*}.bin   --options=\"${copt}\"
            fi
        fi
    done



