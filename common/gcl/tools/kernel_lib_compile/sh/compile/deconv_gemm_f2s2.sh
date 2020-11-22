for file in *
    do
        if [ "${file##*.}"x = "cl"x ];then
            if [[ "${file}" == "deconv_gemm_f2s2.cl" ]];then
                 echo ./gcl_binary --input=$file --output=${file%.*}_12.bin --options=\"${copt} -D ON=1 -D IN=1 -D LN=1 -D KN=2 -DUSE_HALF\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_22.bin --options=\"${copt} -D ON=2 -D IN=2 -D LN=2 -D KN=2 -DUSE_HALF\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_32.bin --options=\"${copt} -D ON=3 -D IN=3 -D LN=3 -D KN=2 -DUSE_HALF\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_42.bin --options=\"${copt} -D ON=4 -D IN=4 -D LN=4 -D KN=2 -DUSE_HALF\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_14.bin --options=\"${copt} -D ON=1 -D IN=1 -D LN=1 -D KN=4 -DUSE_HALF\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_24.bin --options=\"${copt} -D ON=2 -D IN=2 -D LN=2 -D KN=4 -DUSE_HALF\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_34.bin --options=\"${copt} -D ON=3 -D IN=3 -D LN=3 -D KN=4 -DUSE_HALF\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_h_22.bin --options=\"${copt} -D ON=2 -D IN=2 -D LN=2 -D KN=2 -DUSE_HALF -DREUSE_H\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_h_42.bin --options=\"${copt} -D ON=4 -D IN=4 -D LN=4 -D KN=2 -DUSE_HALF -DREUSE_H\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_h_24.bin --options=\"${copt} -D ON=2 -D IN=2 -D LN=2 -D KN=4 -DUSE_HALF -DREUSE_H\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_relu_12.bin --options=\"${copt} -D ON=1 -D IN=1 -D LN=1 -D KN=2 -DUSE_HALF -DUSE_RELU\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_relu_22.bin --options=\"${copt} -D ON=2 -D IN=2 -D LN=2 -D KN=2 -DUSE_HALF -DUSE_RELU\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_relu_32.bin --options=\"${copt} -D ON=3 -D IN=3 -D LN=3 -D KN=2 -DUSE_HALF -DUSE_RELU\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_relu_42.bin --options=\"${copt} -D ON=4 -D IN=4 -D LN=4 -D KN=2 -DUSE_HALF -DUSE_RELU\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_relu_14.bin --options=\"${copt} -D ON=1 -D IN=1 -D LN=1 -D KN=4 -DUSE_HALF -DUSE_RELU\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_relu_24.bin --options=\"${copt} -D ON=2 -D IN=2 -D LN=2 -D KN=4 -DUSE_HALF -DUSE_RELU\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_relu_34.bin --options=\"${copt} -D ON=3 -D IN=3 -D LN=3 -D KN=4 -DUSE_HALF -DUSE_RELU\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_h_relu_22.bin --options=\"${copt} -D ON=2 -D IN=2 -D LN=2 -D KN=2 -DUSE_HALF -DREUSE_H -DUSE_RELU\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_h_relu_42.bin --options=\"${copt} -D ON=4 -D IN=4 -D LN=4 -D KN=2 -DUSE_HALF -DREUSE_H -DUSE_RELU\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_h_relu_24.bin --options=\"${copt} -D ON=2 -D IN=2 -D LN=2 -D KN=4 -DUSE_HALF -DREUSE_H -DUSE_RELU\"
            fi
        fi
    done



