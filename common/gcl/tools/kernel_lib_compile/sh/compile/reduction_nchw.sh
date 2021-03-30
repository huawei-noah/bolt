for file in *
    do
        if [ "${file##*.}"x = "cl"x ];then
            if [[ "${file}" == "reduction_nchw.cl" ]];then
                 echo ./gcl_binary --input=$file --output=${file%.*}_sum0.bin --options=\"${copt} -D AXIS=0 -D TP=sum -DUSE_SUM\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_sum1.bin --options=\"${copt} -D AXIS=1 -D TP=sum -DUSE_SUM\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_sum2.bin --options=\"${copt} -D AXIS=2 -D TP=sum -DUSE_SUM\"

                 echo ./gcl_binary --input=$file --output=${file%.*}_mean0.bin --options=\"${copt} -D AXIS=0 -D TP=mean -DUSE_MEAN\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_mean1.bin --options=\"${copt} -D AXIS=1 -D TP=mean -DUSE_MEAN\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_mean2.bin --options=\"${copt} -D AXIS=2 -D TP=mean -DUSE_MEAN\"

                 echo ./gcl_binary --input=$file --output=${file%.*}_std_deviation0.bin --options=\"${copt} -D AXIS=0 -D TP=std_deviation -DUSE_STD_DEVIATION\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_std_deviation1.bin --options=\"${copt} -D AXIS=1 -D TP=std_deviation -DUSE_STD_DEVIATION\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_std_deviation2.bin --options=\"${copt} -D AXIS=2 -D TP=std_deviation -DUSE_STD_DEVIATION\"

                 echo ./gcl_binary --input=$file --output=${file%.*}_scalar_product0.bin --options=\"${copt} -D AXIS=0 -D TP=scalar_product -DUSE_SCALAR_PRODUCT\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_scalar_product1.bin --options=\"${copt} -D AXIS=1 -D TP=scalar_product -DUSE_SCALAR_PRODUCT\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_scalar_product2.bin --options=\"${copt} -D AXIS=2 -D TP=scalar_product -DUSE_SCALAR_PRODUCT\"
            fi
        fi
    done



