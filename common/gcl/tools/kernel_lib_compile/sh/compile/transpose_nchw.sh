for file in *
    do
        if [ "${file##*.}"x = "cl"x ];then
            if [[ "${file}" == "transpose_nchw.cl" ]];then
                 echo ./gcl_binary --input=$file --output=${file%.*}_0231.bin --options=\"${copt} -D OC=2 -D OH=3 -D OW=1\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_0213.bin --options=\"${copt} -D OC=2 -D OH=1 -D OW=3\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_0312.bin --options=\"${copt} -D OC=3 -D OH=1 -D OW=2\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_0321.bin --options=\"${copt} -D OC=3 -D OH=2 -D OW=1\"
            fi
        fi
    done



