for file in *
    do
        if [ "${file##*.}"x = "cl"x ];then
            if [[ "${file}" == "scale.cl" ]];then
                 echo ./gcl_binary --input=$file --output=${file%.*}_nobeta.bin --options=\"${copt} -D MD=nobeta \"
                 echo ./gcl_binary --input=$file --output=${file%.*}_beta.bin   --options=\"${copt} -D MD=beta -DUSE_BETA\"
            fi
        fi
    done



