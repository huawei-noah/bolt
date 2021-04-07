for file in *
    do
        if [ "${file##*.}"x = "cl"x ];then
            if [[ "${file}" == "conv_wino_rotate_fltbuf.cl" ]];then
                 echo ./gcl_binary --input=$file --output=${file%.*}_3.bin --options=\"${copt} -D F=3 -D Fsq=9\"
            fi
        fi
    done



