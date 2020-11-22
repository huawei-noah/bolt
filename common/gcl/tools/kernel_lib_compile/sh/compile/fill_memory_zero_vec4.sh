for file in *
    do
        if [ "${file##*.}"x = "cl"x ];then
            if [[ "${file}" == "fill_memory_zero_vec4.cl" ]];then
                 echo ./gcl_binary --input=$file --output=${file%.*}_f16.bin --options=\"${copt} -D DT=f16\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_i32.bin --options=\"-D T=int -D T2=int2 -D T3=int3 -D T4=int4 -D DT=i32\"
            fi
        fi
    done



