for file in *
    do
        if [ "${file##*.}"x = "cl"x ];then
            if [[ "${file}" == "fc_trans_fltbuf.cl" ]];then
                echo ./gcl_binary --input=$file --output=${file%.*}_44.bin  --options=\"${copt} -D C=4 -D K=4 -DUSE_HALF\"
            fi
        fi
    done



