for file in *
    do
        if [ "${file##*.}"x = "cl"x ];then
            if [[ "${file}" == "scale.cl" ]];then
                 echo ./gcl_binary --input=$file --output=${file%.*}_nobeta.bin --options=\"${copt} -D MD=nobeta \"
                 echo ./gcl_binary --input=$file --output=${file%.*}_beta.bin   --options=\"${copt} -D MD=beta -DUSE_BETA\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_nchw_nobeta.bin --options=\"${copt} -D MD=nobeta -DUSE_NCHW\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_nchw_beta.bin   --options=\"${copt} -D MD=beta -DUSE_BETA -DUSE_NCHW\"
            fi
        fi
    done



