for file in *
    do
        if [ "${file##*.}"x = "cl"x ];then
            if [[ "${file}" == "eltwise.cl" ]];then
                 echo ./gcl_binary --input=$file --output=${file%.*}_max1.bin --options=\"${copt} -D N=1 -D TP=max -DUSE_MAX\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_max2.bin --options=\"${copt} -D N=2 -D TP=max -DUSE_MAX\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_max3.bin --options=\"${copt} -D N=3 -D TP=max -DUSE_MAX\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_max4.bin --options=\"${copt} -D N=4 -D TP=max -DUSE_MAX\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_max5.bin --options=\"${copt} -D N=5 -D TP=max -DUSE_MAX\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_max6.bin --options=\"${copt} -D N=6 -D TP=max -DUSE_MAX\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_max7.bin --options=\"${copt} -D N=7 -D TP=max -DUSE_MAX\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_max8.bin --options=\"${copt} -D N=8 -D TP=max -DUSE_MAX\"

                 echo ./gcl_binary --input=$file --output=${file%.*}_sum1.bin --options=\"${copt} -D N=1 -D TP=sum -DUSE_SUM\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_sum2.bin --options=\"${copt} -D N=2 -D TP=sum -DUSE_SUM\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_sum3.bin --options=\"${copt} -D N=3 -D TP=sum -DUSE_SUM\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_sum4.bin --options=\"${copt} -D N=4 -D TP=sum -DUSE_SUM\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_sum5.bin --options=\"${copt} -D N=5 -D TP=sum -DUSE_SUM\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_sum6.bin --options=\"${copt} -D N=6 -D TP=sum -DUSE_SUM\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_sum7.bin --options=\"${copt} -D N=7 -D TP=sum -DUSE_SUM\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_sum8.bin --options=\"${copt} -D N=8 -D TP=sum -DUSE_SUM\"

                 echo ./gcl_binary --input=$file --output=${file%.*}_prod1.bin --options=\"${copt} -D N=1 -D TP=prod -DUSE_PROD\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_prod2.bin --options=\"${copt} -D N=2 -D TP=prod -DUSE_PROD\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_prod3.bin --options=\"${copt} -D N=3 -D TP=prod -DUSE_PROD\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_prod4.bin --options=\"${copt} -D N=4 -D TP=prod -DUSE_PROD\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_prod5.bin --options=\"${copt} -D N=5 -D TP=prod -DUSE_PROD\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_prod6.bin --options=\"${copt} -D N=6 -D TP=prod -DUSE_PROD\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_prod7.bin --options=\"${copt} -D N=7 -D TP=prod -DUSE_PROD\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_prod8.bin --options=\"${copt} -D N=8 -D TP=prod -DUSE_PROD\"
            fi
        fi
    done



