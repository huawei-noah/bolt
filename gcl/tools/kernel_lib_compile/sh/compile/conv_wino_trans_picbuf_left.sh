for file in *
    do
        if [ "${file##*.}"x = "cl"x ];then
            if [[ "${file}" == "conv_wino_trans_picbuf_left.cl" ]];then
                 echo ./gcl_binary --input=$file --output=${file%.*}_1.bin --options=\"${copt} -D ON=1 -D UN=0 \"
                 echo ./gcl_binary --input=$file --output=${file%.*}_2.bin --options=\"${copt} -D ON=2 -D UN=1 \"
                 echo ./gcl_binary --input=$file --output=${file%.*}_3.bin --options=\"${copt} -D ON=3 -D UN=2 \"
                 echo ./gcl_binary --input=$file --output=${file%.*}_4.bin --options=\"${copt} -D ON=4 -D UN=3 \"
            fi
        fi
    done



