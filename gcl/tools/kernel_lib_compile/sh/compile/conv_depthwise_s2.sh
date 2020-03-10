for file in *
    do
        if [ "${file##*.}"x = "cl"x ];then
            if [[ "${file}" == "conv_depthwise_s2.cl" ]];then
                echo ./gcl_binary --input=$file --output=${file%.*}_relu_31.bin  --options=\"${copt} -D F=3 -D W=1 -D N=3 -D Fsq=9 -DUSE_HALF -DUSE_RELU\"
                echo ./gcl_binary --input=$file --output=${file%.*}_relu_32.bin  --options=\"${copt} -D F=3 -D W=2 -D N=5 -D Fsq=9 -DUSE_HALF -DUSE_RELU\"
                echo ./gcl_binary --input=$file --output=${file%.*}_relu_33.bin  --options=\"${copt} -D F=3 -D W=3 -D N=7 -D Fsq=9 -DUSE_HALF -DUSE_RELU\"
                echo ./gcl_binary --input=$file --output=${file%.*}_relu_34.bin  --options=\"${copt} -D F=3 -D W=4 -D N=9 -D Fsq=9 -DUSE_HALF -DUSE_RELU\"
                echo ./gcl_binary --input=$file --output=${file%.*}_31.bin  --options=\"${copt} -D F=3 -D W=1 -D N=3 -D Fsq=9 -DUSE_HALF\"
                echo ./gcl_binary --input=$file --output=${file%.*}_32.bin  --options=\"${copt} -D F=3 -D W=2 -D N=5 -D Fsq=9 -DUSE_HALF\"
                echo ./gcl_binary --input=$file --output=${file%.*}_33.bin  --options=\"${copt} -D F=3 -D W=3 -D N=7 -D Fsq=9 -DUSE_HALF\"
                echo ./gcl_binary --input=$file --output=${file%.*}_34.bin  --options=\"${copt} -D F=3 -D W=4 -D N=9 -D Fsq=9 -DUSE_HALF\"

                echo ./gcl_binary --input=$file --output=${file%.*}_relu_51.bin  --options=\"${copt} -D F=5 -D W=1 -D N=5 -D Fsq=25 -DUSE_HALF -DUSE_RELU\"
                echo ./gcl_binary --input=$file --output=${file%.*}_relu_52.bin  --options=\"${copt} -D F=5 -D W=2 -D N=7 -D Fsq=25 -DUSE_HALF -DUSE_RELU\"
                echo ./gcl_binary --input=$file --output=${file%.*}_relu_53.bin  --options=\"${copt} -D F=5 -D W=3 -D N=9 -D Fsq=25 -DUSE_HALF -DUSE_RELU\"
                echo ./gcl_binary --input=$file --output=${file%.*}_relu_54.bin  --options=\"${copt} -D F=5 -D W=4 -D N=11 -D Fsq=25 -DUSE_HALF -DUSE_RELU\"
                echo ./gcl_binary --input=$file --output=${file%.*}_51.bin  --options=\"${copt} -D F=5 -D W=1 -D N=5 -D Fsq=25 -DUSE_HALF\"
                echo ./gcl_binary --input=$file --output=${file%.*}_52.bin  --options=\"${copt} -D F=5 -D W=2 -D N=7 -D Fsq=25 -DUSE_HALF\"
                echo ./gcl_binary --input=$file --output=${file%.*}_53.bin  --options=\"${copt} -D F=5 -D W=3 -D N=9 -D Fsq=25 -DUSE_HALF\"
                echo ./gcl_binary --input=$file --output=${file%.*}_54.bin  --options=\"${copt} -D F=5 -D W=4 -D N=11 -D Fsq=25 -DUSE_HALF\"
            fi
        fi
    done



