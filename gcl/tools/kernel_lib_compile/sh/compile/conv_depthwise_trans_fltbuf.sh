for file in *
    do
        if [ "${file##*.}"x = "cl"x ];then
            if [[ "${file}" == "conv_depthwise_trans_fltbuf.cl" ]];then
                echo ./gcl_binary --input=$file --output=${file%.*}_4.bin  --options=\"${copt} -D K=4 -DUSE_HALF\"
                #echo ./gcl_binary --input=$file --output=${file%.*}_8.bin  --options=\"${copt} -D K=8 -DUSE_HALF\"
            fi
        fi
    done



