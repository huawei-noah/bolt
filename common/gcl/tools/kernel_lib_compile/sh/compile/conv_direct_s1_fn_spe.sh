for file in *
    do
        if [ "${file##*.}"x = "cl"x ];then
            if [[ "${file}" == "conv_direct_s1_fn_spe.cl" ]];then
                 echo ./gcl_binary --input=$file --output=${file%.*}_18.bin       --options=\"${copt} -D F=1 -D ON=8 -D IN=8 -D LN=8 -D Fsq=1 -DUSE_HALF\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_relu_18.bin  --options=\"${copt} -D F=1 -D ON=8 -D IN=8 -D LN=8 -D Fsq=1 -DUSE_RELU -DUSE_HALF\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_relu6_18.bin --options=\"${copt} -D F=1 -D ON=8 -D IN=8 -D LN=8 -D Fsq=1 -DUSE_RELU6 -DUSE_HALF\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_nchw_18.bin       --options=\"${copt} -D F=1 -D ON=8 -D IN=8 -D LN=8 -D Fsq=1 -DUSE_NCHW -DUSE_HALF\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_relu_nchw_18.bin  --options=\"${copt} -D F=1 -D ON=8 -D IN=8 -D LN=8 -D Fsq=1 -DUSE_RELU -DUSE_NCHW -DUSE_HALF\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_relu6_nchw_18.bin --options=\"${copt} -D F=1 -D ON=8 -D IN=8 -D LN=8 -D Fsq=1 -DUSE_RELU6 -DUSE_NCHW -DUSE_HALF\"

                 echo ./gcl_binary --input=$file --output=${file%.*}_28.bin       --options=\"${copt} -D F=2 -D ON=8 -D IN=9 -D LN=9 -D UN=8 -D Fsq=4 -DUSE_HALF\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_relu_28.bin  --options=\"${copt} -D F=2 -D ON=8 -D IN=9 -D LN=9 -D UN=8 -D Fsq=4 -DUSE_RELU -DUSE_HALF\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_relu6_28.bin --options=\"${copt} -D F=2 -D ON=8 -D IN=9 -D LN=9 -D UN=8 -D Fsq=4 -DUSE_RELU6 -DUSE_HALF\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_nchw_28.bin       --options=\"${copt} -D F=2 -D ON=8 -D IN=9 -D LN=9 -D UN=8 -D Fsq=4 -DUSE_NCHW -DUSE_HALF\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_relu_nchw_28.bin  --options=\"${copt} -D F=2 -D ON=8 -D IN=9 -D LN=9 -D UN=8 -D Fsq=4 -DUSE_RELU -DUSE_NCHW -DUSE_HALF\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_relu6_nchw_28.bin --options=\"${copt} -D F=2 -D ON=8 -D IN=9 -D LN=9 -D UN=8 -D Fsq=4 -DUSE_RELU6 -DUSE_NCHW -DUSE_HALF\"

                 echo ./gcl_binary --input=$file --output=${file%.*}_38.bin       --options=\"${copt} -D F=3 -D ON=8 -D IN=10 -D LN=10 -D UN=9 -D Fsq=9 -DUSE_HALF\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_relu_38.bin  --options=\"${copt} -D F=3 -D ON=8 -D IN=10 -D LN=10 -D UN=9 -D Fsq=9 -DUSE_RELU -DUSE_HALF\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_relu6_38.bin --options=\"${copt} -D F=3 -D ON=8 -D IN=10 -D LN=10 -D UN=9 -D Fsq=9 -DUSE_RELU6 -DUSE_HALF\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_nchw_38.bin       --options=\"${copt} -D F=3 -D ON=8 -D IN=10 -D LN=10 -D UN=9 -D Fsq=9 -DUSE_NCHW -DUSE_HALF\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_relu_nchw_38.bin  --options=\"${copt} -D F=3 -D ON=8 -D IN=10 -D LN=10 -D UN=9 -D Fsq=9 -DUSE_RELU -DUSE_NCHW -DUSE_HALF\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_relu6_nchw_38.bin --options=\"${copt} -D F=3 -D ON=8 -D IN=10 -D LN=10 -D UN=9 -D Fsq=9 -DUSE_RELU6 -DUSE_NCHW -DUSE_HALF\"

                 echo ./gcl_binary --input=$file --output=${file%.*}_58.bin       --options=\"${copt} -D F=5 -D ON=8 -D IN=12 -D LN=12 -D UN=11 -D Fsq=25 -DUSE_HALF\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_relu_58.bin  --options=\"${copt} -D F=5 -D ON=8 -D IN=12 -D LN=12 -D UN=11 -D Fsq=25 -DUSE_RELU -DUSE_HALF\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_relu6_58.bin --options=\"${copt} -D F=5 -D ON=8 -D IN=12 -D LN=12 -D UN=11 -D Fsq=25 -DUSE_RELU6 -DUSE_HALF\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_nchw_58.bin       --options=\"${copt} -D F=5 -D ON=8 -D IN=12 -D LN=12 -D UN=11 -D Fsq=25 -DUSE_NCHW -DUSE_HALF\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_relu_nchw_58.bin  --options=\"${copt} -D F=5 -D ON=8 -D IN=12 -D LN=12 -D UN=11 -D Fsq=25 -DUSE_RELU -DUSE_NCHW -DUSE_HALF\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_relu6_nchw_58.bin --options=\"${copt} -D F=5 -D ON=8 -D IN=12 -D LN=12 -D UN=11 -D Fsq=25 -DUSE_RELU6 -DUSE_NCHW -DUSE_HALF\"

                 echo ./gcl_binary --input=$file --output=${file%.*}_76.bin       --options=\"${copt} -D F=7 -D ON=6 -D IN=12 -D LN=12 -D UN=11 -D Fsq=49 -DUSE_HALF\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_relu_76.bin  --options=\"${copt} -D F=7 -D ON=6 -D IN=12 -D LN=12 -D UN=11 -D Fsq=49 -DUSE_RELU -DUSE_HALF\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_relu6_76.bin --options=\"${copt} -D F=7 -D ON=6 -D IN=12 -D LN=12 -D UN=11 -D Fsq=49 -DUSE_RELU6 -DUSE_HALF\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_nchw_76.bin       --options=\"${copt} -D F=7 -D ON=6 -D IN=12 -D LN=12 -D UN=11 -D Fsq=49 -DUSE_NCHW -DUSE_HALF\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_relu_nchw_76.bin  --options=\"${copt} -D F=7 -D ON=6 -D IN=12 -D LN=12 -D UN=11 -D Fsq=49 -DUSE_RELU -DUSE_NCHW -DUSE_HALF\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_relu6_nchw_76.bin --options=\"${copt} -D F=7 -D ON=6 -D IN=12 -D LN=12 -D UN=11 -D Fsq=49 -DUSE_RELU6 -DUSE_NCHW -DUSE_HALF\"
            fi
        fi
    done



