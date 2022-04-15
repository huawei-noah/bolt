for file in *
    do
        if [ "${file##*.}"x = "cl"x ];then
            if [[ "${file}" == "topk_sort.cl" ]];then
                 echo ./gcl_binary --input=$file --output=${file%.*}_max.bin --options=\"${copt} -D USE_MAX\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_min.bin --options=\"${copt} -D USE_MIN\"
            fi
        fi
    done



