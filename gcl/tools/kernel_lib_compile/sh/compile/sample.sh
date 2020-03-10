for file in *
    do
        if [ "${file##*.}"x = "cl"x ];then
             if [[ "${file}" == sample.cl ]];then
                echo ./gcl_binary --input=$file --output=${file%.*}.bin  --options=\"${copt} -D C=12\"
             fi
        fi
    done



