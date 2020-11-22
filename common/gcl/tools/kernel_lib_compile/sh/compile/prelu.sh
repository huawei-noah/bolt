for file in *
    do
        if [ "${file##*.}"x = "cl"x ];then
            if [[ "${file}" == "prelu.cl" ]];then
                 echo ./gcl_binary --input=$file --output=${file%.*}_noprop.bin --options=\"${copt} -D MD=noprop \"
                 echo ./gcl_binary --input=$file --output=${file%.*}_prop.bin --options=\"${copt} -D MD=prop -DUSE_SAME \"
            fi
        fi
    done



