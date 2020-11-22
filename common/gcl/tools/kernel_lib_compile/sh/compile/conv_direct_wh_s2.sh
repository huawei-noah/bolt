for file in *
    do
        if [ "${file##*.}"x = "cl"x ];then
            if [[ "${file}" == "conv_direct_wh_s2.cl" ]];then
                # W=3 H=1 Stride = 2
                echo ./gcl_binary --input=$file --output=${file%.*}_3111.bin --options=\"${copt} -D W=3 -D H=1 -D ON=1 -D IN=3 -D LN=3 -D UN=2 -D Fsq=3 -D KN=1 -DUSE_HALF\"
                echo ./gcl_binary --input=$file --output=${file%.*}_3121.bin --options=\"${copt} -D W=3 -D H=1 -D ON=2 -D IN=2 -D LN=1 -D UN=1 -D Fsq=3 -D KN=1 -DUSE_HALF -DBASIC_REG\"
                echo ./gcl_binary --input=$file --output=${file%.*}_3131.bin --options=\"${copt} -D W=3 -D H=1 -D ON=3 -D IN=3 -D LN=2 -D UN=2 -D Fsq=3 -D KN=1 -DUSE_HALF -DBASIC_REG\"
                echo ./gcl_binary --input=$file --output=${file%.*}_3141.bin --options=\"${copt} -D W=3 -D H=1 -D ON=4 -D IN=4 -D LN=3 -D UN=3 -D Fsq=3 -D KN=1 -DUSE_HALF -DBASIC_REG\"
                echo ./gcl_binary --input=$file --output=${file%.*}_3151.bin --options=\"${copt} -D W=3 -D H=1 -D ON=5 -D IN=5 -D LN=4 -D UN=4 -D Fsq=3 -D KN=1 -DUSE_HALF -DBASIC_REG\"
                echo ./gcl_binary --input=$file --output=${file%.*}_3161.bin --options=\"${copt} -D W=3 -D H=1 -D ON=6 -D IN=6 -D LN=5 -D UN=5 -D Fsq=3 -D KN=1 -DUSE_HALF -DBASIC_REG\"
                echo ./gcl_binary --input=$file --output=${file%.*}_3171.bin --options=\"${copt} -D W=3 -D H=1 -D ON=7 -D IN=7 -D LN=6 -D UN=6 -D Fsq=3 -D KN=1 -DUSE_HALF -DBASIC_REG\"
                echo ./gcl_binary --input=$file --output=${file%.*}_3181.bin --options=\"${copt} -D W=3 -D H=1 -D ON=8 -D IN=8 -D LN=7 -D UN=7 -D Fsq=3 -D KN=1 -DUSE_HALF -DBASIC_REG\"

                echo ./gcl_binary --input=$file --output=${file%.*}_relu_3111.bin --options=\"${copt} -D W=3 -D H=1 -D ON=1 -D IN=3 -D LN=3 -D UN=2 -D Fsq=3 -D KN=1 -DUSE_HALF -DUSE_RELU\"
                echo ./gcl_binary --input=$file --output=${file%.*}_relu_3121.bin --options=\"${copt} -D W=3 -D H=1 -D ON=2 -D IN=2 -D LN=1 -D UN=1 -D Fsq=3 -D KN=1 -DUSE_HALF -DUSE_RELU -DBASIC_REG\"
                echo ./gcl_binary --input=$file --output=${file%.*}_relu_3131.bin --options=\"${copt} -D W=3 -D H=1 -D ON=3 -D IN=3 -D LN=2 -D UN=2 -D Fsq=3 -D KN=1 -DUSE_HALF -DUSE_RELU -DBASIC_REG\"
                echo ./gcl_binary --input=$file --output=${file%.*}_relu_3141.bin --options=\"${copt} -D W=3 -D H=1 -D ON=4 -D IN=4 -D LN=3 -D UN=3 -D Fsq=3 -D KN=1 -DUSE_HALF -DUSE_RELU -DBASIC_REG\"
                echo ./gcl_binary --input=$file --output=${file%.*}_relu_3151.bin --options=\"${copt} -D W=3 -D H=1 -D ON=5 -D IN=5 -D LN=4 -D UN=4 -D Fsq=3 -D KN=1 -DUSE_HALF -DUSE_RELU -DBASIC_REG\"
                echo ./gcl_binary --input=$file --output=${file%.*}_relu_3161.bin --options=\"${copt} -D W=3 -D H=1 -D ON=6 -D IN=6 -D LN=5 -D UN=5 -D Fsq=3 -D KN=1 -DUSE_HALF -DUSE_RELU -DBASIC_REG\"
                echo ./gcl_binary --input=$file --output=${file%.*}_relu_3171.bin --options=\"${copt} -D W=3 -D H=1 -D ON=7 -D IN=7 -D LN=6 -D UN=6 -D Fsq=3 -D KN=1 -DUSE_HALF -DUSE_RELU -DBASIC_REG\"
                echo ./gcl_binary --input=$file --output=${file%.*}_relu_3181.bin --options=\"${copt} -D W=3 -D H=1 -D ON=8 -D IN=8 -D LN=7 -D UN=7 -D Fsq=3 -D KN=1 -DUSE_HALF -DUSE_RELU -DBASIC_REG\"

                echo ./gcl_binary --input=$file --output=${file%.*}_relu6_3111.bin --options=\"${copt} -D W=3 -D H=1 -D ON=1 -D IN=3 -D LN=3 -D UN=2 -D Fsq=3 -D KN=1 -DUSE_HALF -DUSE_RELU6\"
                echo ./gcl_binary --input=$file --output=${file%.*}_relu6_3121.bin --options=\"${copt} -D W=3 -D H=1 -D ON=2 -D IN=2 -D LN=1 -D UN=1 -D Fsq=3 -D KN=1 -DUSE_HALF -DUSE_RELU6 -DBASIC_REG\"
                echo ./gcl_binary --input=$file --output=${file%.*}_relu6_3131.bin --options=\"${copt} -D W=3 -D H=1 -D ON=3 -D IN=3 -D LN=2 -D UN=2 -D Fsq=3 -D KN=1 -DUSE_HALF -DUSE_RELU6 -DBASIC_REG\"
                echo ./gcl_binary --input=$file --output=${file%.*}_relu6_3141.bin --options=\"${copt} -D W=3 -D H=1 -D ON=4 -D IN=4 -D LN=3 -D UN=3 -D Fsq=3 -D KN=1 -DUSE_HALF -DUSE_RELU6 -DBASIC_REG\"
                echo ./gcl_binary --input=$file --output=${file%.*}_relu6_3151.bin --options=\"${copt} -D W=3 -D H=1 -D ON=5 -D IN=5 -D LN=4 -D UN=4 -D Fsq=3 -D KN=1 -DUSE_HALF -DUSE_RELU6 -DBASIC_REG\"
                echo ./gcl_binary --input=$file --output=${file%.*}_relu6_3161.bin --options=\"${copt} -D W=3 -D H=1 -D ON=6 -D IN=6 -D LN=5 -D UN=5 -D Fsq=3 -D KN=1 -DUSE_HALF -DUSE_RELU6 -DBASIC_REG\"
                echo ./gcl_binary --input=$file --output=${file%.*}_relu6_3171.bin --options=\"${copt} -D W=3 -D H=1 -D ON=7 -D IN=7 -D LN=6 -D UN=6 -D Fsq=3 -D KN=1 -DUSE_HALF -DUSE_RELU6 -DBASIC_REG\"
                echo ./gcl_binary --input=$file --output=${file%.*}_relu6_3181.bin --options=\"${copt} -D W=3 -D H=1 -D ON=8 -D IN=8 -D LN=7 -D UN=7 -D Fsq=3 -D KN=1 -DUSE_HALF -DUSE_RELU6 -DBASIC_REG\"

                echo ./gcl_binary --input=$file --output=${file%.*}_3112.bin --options=\"${copt} -D W=3 -D H=1 -D ON=1 -D IN=3 -D LN=3 -D UN=2 -D Fsq=3 -D KN=2 -DUSE_HALF\"
                echo ./gcl_binary --input=$file --output=${file%.*}_3122.bin --options=\"${copt} -D W=3 -D H=1 -D ON=2 -D IN=5 -D LN=5 -D UN=4 -D Fsq=3 -D KN=2 -DUSE_HALF\"
                echo ./gcl_binary --input=$file --output=${file%.*}_3132.bin --options=\"${copt} -D W=3 -D H=1 -D ON=3 -D IN=7 -D LN=7 -D UN=6 -D Fsq=3 -D KN=2 -DUSE_HALF\"
                echo ./gcl_binary --input=$file --output=${file%.*}_3142.bin --options=\"${copt} -D W=3 -D H=1 -D ON=4 -D IN=4 -D LN=3 -D UN=3 -D Fsq=3 -D KN=2 -DUSE_HALF -DBASIC_REG\"

                echo ./gcl_binary --input=$file --output=${file%.*}_relu_3112.bin --options=\"${copt} -D W=3 -D H=1 -D ON=1 -D IN=3 -D LN=3 -D UN=2 -D Fsq=3 -D KN=2 -DUSE_HALF -DUSE_RELU\"
                echo ./gcl_binary --input=$file --output=${file%.*}_relu_3122.bin --options=\"${copt} -D W=3 -D H=1 -D ON=2 -D IN=5 -D LN=5 -D UN=4 -D Fsq=3 -D KN=2 -DUSE_HALF -DUSE_RELU\"
                echo ./gcl_binary --input=$file --output=${file%.*}_relu_3132.bin --options=\"${copt} -D W=3 -D H=1 -D ON=3 -D IN=7 -D LN=7 -D UN=6 -D Fsq=3 -D KN=2 -DUSE_HALF -DUSE_RELU\"
                echo ./gcl_binary --input=$file --output=${file%.*}_relu_3142.bin --options=\"${copt} -D W=3 -D H=1 -D ON=4 -D IN=4 -D LN=3 -D UN=3 -D Fsq=3 -D KN=2 -DUSE_HALF -DUSE_RELU -DBASIC_REG\"

                echo ./gcl_binary --input=$file --output=${file%.*}_relu6_3112.bin --options=\"${copt} -D W=3 -D H=1 -D ON=1 -D IN=3 -D LN=3 -D UN=2 -D Fsq=3 -D KN=2 -DUSE_HALF -DUSE_RELU6\"
                echo ./gcl_binary --input=$file --output=${file%.*}_relu6_3122.bin --options=\"${copt} -D W=3 -D H=1 -D ON=2 -D IN=5 -D LN=5 -D UN=4 -D Fsq=3 -D KN=2 -DUSE_HALF -DUSE_RELU6\"
                echo ./gcl_binary --input=$file --output=${file%.*}_relu6_3132.bin --options=\"${copt} -D W=3 -D H=1 -D ON=3 -D IN=7 -D LN=7 -D UN=6 -D Fsq=3 -D KN=2 -DUSE_HALF -DUSE_RELU6\"
                echo ./gcl_binary --input=$file --output=${file%.*}_relu6_3142.bin --options=\"${copt} -D W=3 -D H=1 -D ON=4 -D IN=4 -D LN=3 -D UN=3 -D Fsq=3 -D KN=2 -DUSE_HALF -DUSE_RELU6 -DBASIC_REG\"
            fi
        fi
    done





