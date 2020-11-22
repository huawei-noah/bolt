for file in *
    do
        if [ "${file##*.}"x = "cl"x ];then
            if [[ "${file}" == "fill_memory_zero.cl" ]];then
                 echo ./gcl_binary --input=$file --output=${file%.*}_f16.bin --options=\"${copt} -D DT=f16\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_i32.bin --options=\"-D T=int -D DT=i32\"
            fi
        fi
    done



