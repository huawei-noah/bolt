for file in *
    do
        if [ "${file##*.}"x = "cl"x ];then
            if [[ "${file}" == "conv_wino_trans_picbuf_right.cl" ]];then
                 echo ./gcl_binary --input=$file --output=${file%.*}.bin --options=\"${copt} -D ON=4 -D UN=3\"
            fi
        fi
    done



