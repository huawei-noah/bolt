for file in *
    do
        if [ "${file##*.}"x = "cl"x ];then
            if [[ "${file}" == "conv_direct_s1.cl" ]];then
                 echo ./gcl_binary --input=$file --output=${file%.*}_11.bin --options=\"${copt} -D F=1 -D W=1 -D N=1 -D Fsq=1 -DUSE_HALF\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_12.bin --options=\"${copt} -D F=1 -D W=2 -D N=2 -D Fsq=1 -DUSE_HALF\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_13.bin --options=\"${copt} -D F=1 -D W=3 -D N=3 -D Fsq=1 -DUSE_HALF\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_14.bin --options=\"${copt} -D F=1 -D W=4 -D N=4 -D Fsq=1 -DUSE_HALF\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_15.bin --options=\"${copt} -D F=1 -D W=5 -D N=5 -D Fsq=1 -DUSE_HALF\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_16.bin --options=\"${copt} -D F=1 -D W=6 -D N=6 -D Fsq=1 -DUSE_HALF\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_17.bin --options=\"${copt} -D F=1 -D W=7 -D N=7 -D Fsq=1 -DUSE_HALF\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_18.bin --options=\"${copt} -D F=1 -D W=8 -D N=8 -D Fsq=1 -DUSE_HALF\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_relu_11.bin --options=\"${copt} -D F=1 -D W=1 -D N=1 -D Fsq=1 -DUSE_HALF -DUSE_RELU\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_relu_12.bin --options=\"${copt} -D F=1 -D W=2 -D N=2 -D Fsq=1 -DUSE_HALF -DUSE_RELU\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_relu_13.bin --options=\"${copt} -D F=1 -D W=3 -D N=3 -D Fsq=1 -DUSE_HALF -DUSE_RELU\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_relu_14.bin --options=\"${copt} -D F=1 -D W=4 -D N=4 -D Fsq=1 -DUSE_HALF -DUSE_RELU\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_relu_15.bin --options=\"${copt} -D F=1 -D W=5 -D N=5 -D Fsq=1 -DUSE_HALF -DUSE_RELU\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_relu_16.bin --options=\"${copt} -D F=1 -D W=6 -D N=6 -D Fsq=1 -DUSE_HALF -DUSE_RELU\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_relu_17.bin --options=\"${copt} -D F=1 -D W=7 -D N=7 -D Fsq=1 -DUSE_HALF -DUSE_RELU\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_relu_18.bin --options=\"${copt} -D F=1 -D W=8 -D N=8 -D Fsq=1 -DUSE_HALF -DUSE_RELU\"

                 echo ./gcl_binary --input=$file --output=${file%.*}_relu_31.bin  --options=\"${copt} -D F=3 -D W=1  -D N=3   -D Fsq=9  -DUSE_HALF -DUSE_RELU\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_relu_32.bin  --options=\"${copt} -D F=3 -D W=2  -D N=4   -D Fsq=9  -DUSE_HALF -DUSE_RELU\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_relu_33.bin  --options=\"${copt} -D F=3 -D W=3  -D N=5   -D Fsq=9  -DUSE_HALF -DUSE_RELU\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_relu_34.bin  --options=\"${copt} -D F=3 -D W=4  -D N=6   -D Fsq=9  -DUSE_HALF -DUSE_RELU\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_relu_35.bin  --options=\"${copt} -D F=3 -D W=5  -D N=7   -D Fsq=9  -DUSE_HALF -DUSE_RELU\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_relu_36.bin  --options=\"${copt} -D F=3 -D W=6  -D N=8   -D Fsq=9  -DUSE_HALF -DUSE_RELU\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_relu_37.bin  --options=\"${copt} -D F=3 -D W=7  -D N=9   -D Fsq=9  -DUSE_HALF -DUSE_RELU\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_relu_38.bin  --options=\"${copt} -D F=3 -D W=8  -D N=10  -D Fsq=9  -DUSE_HALF -DUSE_RELU\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_31.bin  --options=\"${copt} -D F=3 -D W=1  -D N=3   -D Fsq=9  -DUSE_HALF\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_32.bin  --options=\"${copt} -D F=3 -D W=2  -D N=4   -D Fsq=9  -DUSE_HALF\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_33.bin  --options=\"${copt} -D F=3 -D W=3  -D N=5   -D Fsq=9  -DUSE_HALF\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_34.bin  --options=\"${copt} -D F=3 -D W=4  -D N=6   -D Fsq=9  -DUSE_HALF\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_35.bin  --options=\"${copt} -D F=3 -D W=5  -D N=7   -D Fsq=9  -DUSE_HALF\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_36.bin  --options=\"${copt} -D F=3 -D W=6  -D N=8   -D Fsq=9  -DUSE_HALF\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_37.bin  --options=\"${copt} -D F=3 -D W=7  -D N=9   -D Fsq=9  -DUSE_HALF\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_38.bin  --options=\"${copt} -D F=3 -D W=8  -D N=10  -D Fsq=9  -DUSE_HALF\"

                 echo ./gcl_binary --input=$file --output=${file%.*}_relu_51.bin  --options=\"${copt} -D F=5 -D W=1  -D N=5 -D Fsq=25 -DUSE_HALF\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_relu_52.bin  --options=\"${copt} -D F=5 -D W=2  -D N=6 -D Fsq=25 -DUSE_HALF\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_relu_53.bin  --options=\"${copt} -D F=5 -D W=3  -D N=7 -D Fsq=25 -DUSE_HALF\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_relu_54.bin  --options=\"${copt} -D F=5 -D W=4  -D N=8 -D Fsq=25 -DUSE_HALF\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_51.bin  --options=\"${copt} -D F=5 -D W=1  -D N=5 -D Fsq=25 -DUSE_HALF\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_52.bin  --options=\"${copt} -D F=5 -D W=2  -D N=6 -D Fsq=25 -DUSE_HALF\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_53.bin  --options=\"${copt} -D F=5 -D W=3  -D N=7 -D Fsq=25 -DUSE_HALF\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_54.bin  --options=\"${copt} -D F=5 -D W=4  -D N=8 -D Fsq=25 -DUSE_HALF\"
               # echo ./gcl_binary --input=$file --output=${file%.*}_55.bin  --options=\"-D T=half -D T2=half2 -D T3=half3 -D T4=half4 -D T8=half8 -D T16=half16 -D F=5 -D W=5  -D N=9   -D Wsq=25 -DUSE_HALF\"
               # echo ./gcl_binary --input=$file --output=${file%.*}_56.bin  --options=\"-D T=half -D T2=half2 -D T3=half3 -D T4=half4 -D T8=half8 -D T16=half16 -D F=5 -D W=6  -D N=10  -D Wsq=25 -DUSE_HALF\"
               # echo ./gcl_binary --input=$file --output=${file%.*}_57.bin  --options=\"-D T=half -D T2=half2 -D T3=half3 -D T4=half4 -D T8=half8 -D T16=half16 -D F=5 -D W=7  -D N=11  -D Wsq=25 -DUSE_HALF\"
               # echo ./gcl_binary --input=$file --output=${file%.*}_58.bin  --options=\"-D T=half -D T2=half2 -D T3=half3 -D T4=half4 -D T8=half8 -D T16=half16 -D F=5 -D W=8  -D N=12  -D Wsq=25 -DUSE_HALF\"
               # echo ./gcl_binary --input=$file --output=${file%.*}_510.bin --options=\"-D T=half -D T2=half2 -D T3=half3 -D T4=half4 -D T8=half8 -D T16=half16 -D F=5 -D W=10 -D N=14  -D Wsq=25 -DUSE_HALF\"
            fi
        fi
    done



