for file in *
    do
        if [ "${file##*.}"x = "cl"x ];then
            if [[ "${file}" == "eltwise_broadcast.cl" ]];then
                 echo ./gcl_binary --input=$file --output=${file%.*}_max0.bin --options=\"${copt} -D N=0 -D TP=max -DUSE_MAX\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_max1.bin --options=\"${copt} -D N=1 -D TP=max -DUSE_MAX\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_max2.bin --options=\"${copt} -D N=2 -D TP=max -DUSE_MAX\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_max3.bin --options=\"${copt} -D N=3 -D TP=max -DUSE_MAX\"

                 echo ./gcl_binary --input=$file --output=${file%.*}_sum0.bin --options=\"${copt} -D N=0 -D TP=sum -DUSE_SUM\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_sum1.bin --options=\"${copt} -D N=1 -D TP=sum -DUSE_SUM\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_sum2.bin --options=\"${copt} -D N=2 -D TP=sum -DUSE_SUM\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_sum3.bin --options=\"${copt} -D N=3 -D TP=sum -DUSE_SUM\"

                 echo ./gcl_binary --input=$file --output=${file%.*}_prod0.bin --options=\"${copt} -D N=0 -D TP=prod -DUSE_PROD\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_prod1.bin --options=\"${copt} -D N=1 -D TP=prod -DUSE_PROD\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_prod2.bin --options=\"${copt} -D N=2 -D TP=prod -DUSE_PROD\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_prod3.bin --options=\"${copt} -D N=3 -D TP=prod -DUSE_PROD\"
            fi
        fi
    done

