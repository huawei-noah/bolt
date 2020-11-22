for file in *
    do
        if [ "${file##*.}"x = "cl"x ];then
            if [[ "${file}" == "eltwise.cl" ]];then
                 echo ./gcl_binary --input=$file --output=${file%.*}_max1.bin --options=\"${copt} -D N=1 -D TP=max -DUSE_MAX\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_max2.bin --options=\"${copt} -D N=2 -D TP=max -DUSE_MAX\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_max3.bin --options=\"${copt} -D N=3 -D TP=max -DUSE_MAX\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_max4.bin --options=\"${copt} -D N=4 -D TP=max -DUSE_MAX\"

                 echo ./gcl_binary --input=$file --output=${file%.*}_sum1.bin --options=\"${copt} -D N=1 -D TP=sum -DUSE_SUM\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_sum2.bin --options=\"${copt} -D N=2 -D TP=sum -DUSE_SUM\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_sum3.bin --options=\"${copt} -D N=3 -D TP=sum -DUSE_SUM\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_sum4.bin --options=\"${copt} -D N=4 -D TP=sum -DUSE_SUM\"

                 echo ./gcl_binary --input=$file --output=${file%.*}_prod1.bin --options=\"${copt} -D N=1 -D TP=prod -DUSE_PROD\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_prod2.bin --options=\"${copt} -D N=2 -D TP=prod -DUSE_PROD\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_prod3.bin --options=\"${copt} -D N=3 -D TP=prod -DUSE_PROD\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_prod4.bin --options=\"${copt} -D N=4 -D TP=prod -DUSE_PROD\"

                 echo ./gcl_binary --input=$file --output=${file%.*}_relu_max1.bin --options=\"${copt} -D N=1 -D TP=max -DUSE_MAX -DUSE_RELU\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_relu_max2.bin --options=\"${copt} -D N=2 -D TP=max -DUSE_MAX -DUSE_RELU\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_relu_max3.bin --options=\"${copt} -D N=3 -D TP=max -DUSE_MAX -DUSE_RELU\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_relu_max4.bin --options=\"${copt} -D N=4 -D TP=max -DUSE_MAX -DUSE_RELU\"

                 echo ./gcl_binary --input=$file --output=${file%.*}_relu_sum1.bin --options=\"${copt} -D N=1 -D TP=sum -DUSE_SUM -DUSE_RELU\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_relu_sum2.bin --options=\"${copt} -D N=2 -D TP=sum -DUSE_SUM -DUSE_RELU\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_relu_sum3.bin --options=\"${copt} -D N=3 -D TP=sum -DUSE_SUM -DUSE_RELU\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_relu_sum4.bin --options=\"${copt} -D N=4 -D TP=sum -DUSE_SUM -DUSE_RELU\"

                 echo ./gcl_binary --input=$file --output=${file%.*}_relu_prod1.bin --options=\"${copt} -D N=1 -D TP=prod -DUSE_PROD -DUSE_RELU\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_relu_prod2.bin --options=\"${copt} -D N=2 -D TP=prod -DUSE_PROD -DUSE_RELU\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_relu_prod3.bin --options=\"${copt} -D N=3 -D TP=prod -DUSE_PROD -DUSE_RELU\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_relu_prod4.bin --options=\"${copt} -D N=4 -D TP=prod -DUSE_PROD -DUSE_RELU\"

                 echo ./gcl_binary --input=$file --output=${file%.*}_nchw_max1.bin --options=\"${copt} -D N=1 -D TP=max -DUSE_MAX -DUSE_NCHW\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_nchw_max2.bin --options=\"${copt} -D N=2 -D TP=max -DUSE_MAX -DUSE_NCHW\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_nchw_max3.bin --options=\"${copt} -D N=3 -D TP=max -DUSE_MAX -DUSE_NCHW\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_nchw_max4.bin --options=\"${copt} -D N=4 -D TP=max -DUSE_MAX -DUSE_NCHW\"

                 echo ./gcl_binary --input=$file --output=${file%.*}_nchw_sum1.bin --options=\"${copt} -D N=1 -D TP=sum -DUSE_SUM -DUSE_NCHW\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_nchw_sum2.bin --options=\"${copt} -D N=2 -D TP=sum -DUSE_SUM -DUSE_NCHW\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_nchw_sum3.bin --options=\"${copt} -D N=3 -D TP=sum -DUSE_SUM -DUSE_NCHW\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_nchw_sum4.bin --options=\"${copt} -D N=4 -D TP=sum -DUSE_SUM -DUSE_NCHW\"

                 echo ./gcl_binary --input=$file --output=${file%.*}_nchw_prod1.bin --options=\"${copt} -D N=1 -D TP=prod -DUSE_PROD -DUSE_NCHW\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_nchw_prod2.bin --options=\"${copt} -D N=2 -D TP=prod -DUSE_PROD -DUSE_NCHW\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_nchw_prod3.bin --options=\"${copt} -D N=3 -D TP=prod -DUSE_PROD -DUSE_NCHW\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_nchw_prod4.bin --options=\"${copt} -D N=4 -D TP=prod -DUSE_PROD -DUSE_NCHW\"

                 echo ./gcl_binary --input=$file --output=${file%.*}_nchw_relu_max1.bin --options=\"${copt} -D N=1 -D TP=max -DUSE_MAX -DUSE_RELU -DUSE_NCHW\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_nchw_relu_max2.bin --options=\"${copt} -D N=2 -D TP=max -DUSE_MAX -DUSE_RELU -DUSE_NCHW\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_nchw_relu_max3.bin --options=\"${copt} -D N=3 -D TP=max -DUSE_MAX -DUSE_RELU -DUSE_NCHW\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_nchw_relu_max4.bin --options=\"${copt} -D N=4 -D TP=max -DUSE_MAX -DUSE_RELU -DUSE_NCHW\"

                 echo ./gcl_binary --input=$file --output=${file%.*}_nchw_relu_sum1.bin --options=\"${copt} -D N=1 -D TP=sum -DUSE_SUM -DUSE_RELU -DUSE_NCHW\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_nchw_relu_sum2.bin --options=\"${copt} -D N=2 -D TP=sum -DUSE_SUM -DUSE_RELU -DUSE_NCHW\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_nchw_relu_sum3.bin --options=\"${copt} -D N=3 -D TP=sum -DUSE_SUM -DUSE_RELU -DUSE_NCHW\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_nchw_relu_sum4.bin --options=\"${copt} -D N=4 -D TP=sum -DUSE_SUM -DUSE_RELU -DUSE_NCHW\"

                 echo ./gcl_binary --input=$file --output=${file%.*}_nchw_relu_prod1.bin --options=\"${copt} -D N=1 -D TP=prod -DUSE_PROD -DUSE_RELU -DUSE_NCHW\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_nchw_relu_prod2.bin --options=\"${copt} -D N=2 -D TP=prod -DUSE_PROD -DUSE_RELU -DUSE_NCHW\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_nchw_relu_prod3.bin --options=\"${copt} -D N=3 -D TP=prod -DUSE_PROD -DUSE_RELU -DUSE_NCHW\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_nchw_relu_prod4.bin --options=\"${copt} -D N=4 -D TP=prod -DUSE_PROD -DUSE_RELU -DUSE_NCHW\"
            fi
        fi
    done



