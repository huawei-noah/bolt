for file in *
    do
        if [ "${file##*.}"x = "cl"x ];then
            if [[ "${file}" == "conv_direct_s1_spe_f1c3k1.cl" ]];then
                echo ./gcl_binary --input=$file --output=${file%.*}_0.bin  --options=\"${copt} -D EW=0 -DUSE_HALF\"
                echo ./gcl_binary --input=$file --output=${file%.*}_1.bin  --options=\"${copt} -D EW=1 -DUSE_HALF\"
            fi
        fi
    done



