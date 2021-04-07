for file in *
    do
        if [ "${file##*.}"x = "cl"x ];then
            if [[ "${file}" == "cast.cl" ]];then
                 echo ./gcl_binary --input=$file --output=${file%.*}_f16_to_f16.bin      --options=\"${copt} -D INPUT_F16 -D OUTPUT_F16\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_f16_to_i32.bin      --options=\"${copt} -D INPUT_F16 -D OUTPUT_I32\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_i32_to_i32.bin      --options=\"${copt} -D INPUT_I32 -D OUTPUT_I32\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_i32_to_f16.bin      --options=\"${copt} -D INPUT_I32 -D OUTPUT_F16\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_f16_to_f16_nchw.bin --options=\"${copt} -D INPUT_F16 -D OUTPUT_F16 -D USE_NCHW\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_f16_to_i32_nchw.bin --options=\"${copt} -D INPUT_F16 -D OUTPUT_I32 -D USE_NCHW\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_i32_to_i32_nchw.bin --options=\"${copt} -D INPUT_I32 -D OUTPUT_I32 -D USE_NCHW\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_i32_to_f16_nchw.bin --options=\"${copt} -D INPUT_I32 -D OUTPUT_F16 -D USE_NCHW\"
            fi
        fi
    done



