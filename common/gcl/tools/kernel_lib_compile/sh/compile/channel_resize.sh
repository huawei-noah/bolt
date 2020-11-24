for file in *
    do
        if [ "${file##*.}"x = "cl"x ];then
            if [[ "${file}" == "channel_resize.cl" ]];then
                 echo ./gcl_binary --input=$file --output=${file%.*}.bin       --options=\"${copt}\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_nchw.bin --options=\"${copt} -DINPUT_NCHW -DOUTPUT_NCHW\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_nchw_ncwhc4.bin --options=\"${copt} -DINPUT_NCHW -DOUTPUT_NCWHC4\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_ncwhc4_nchw.bin --options=\"${copt} -DINPUT_NCWHC4 -DOUTPUT_NCHW\"
            fi
        fi
    done



