for file in *
    do
        if [ "${file##*.}"x = "cl"x ];then
            if [[ "${file}" == "padding.cl" ]];then
                 echo ./gcl_binary --input=$file --output=${file%.*}_constant.bin --options=\"${copt} -D USE_CONSTANT\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_edge.bin --options=\"${copt} -D USE_EDGE\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_reflect.bin --options=\"${copt} -D USE_REFLECT\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_symmetric.bin --options=\"${copt} -D USE_SYMMETRIC\"
            fi
        fi
    done



