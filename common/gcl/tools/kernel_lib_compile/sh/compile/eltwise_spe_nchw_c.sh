for file in *
    do
        if [ "${file##*.}"x = "cl"x ];then
            if [[ "${file}" == "eltwise_spe_nchw_c.cl" ]];then
                 echo ./gcl_binary --input=$file --output=${file%.*}_max.bin --options=\"${copt}  -D TP=max -DUSE_MAX\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_sum.bin --options=\"${copt}  -D TP=sum -DUSE_SUM\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_prod.bin --options=\"${copt} -D TP=prod -DUSE_PROD\"
            fi
        fi
    done



