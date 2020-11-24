for file in *
    do
        if [ "${file##*.}"x = "cl"x ];then
            if [[ "${file}" == "copy.cl" ]];then
                 echo ./gcl_binary --input=$file --output=${file%.*}_f16.bin --options=\"${copt} -D DT=f16\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_i32.bin --options=\"-D T=int -D T2=int2 -D T3=int3 -D T4=int4 -D DT=i32\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_u32.bin --options=\"-D T=uint -D T2=uint2 -D T3=uint3 -D T4=uint4 -D DT=u32\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_with_block_index_f16.bin --options=\"${copt}   -D DT=f16 -D USE_BLOCK_INDEX\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_with_block_index_i32.bin --options=\"-D T=int  -D T2=int2 -D T3=int3 -D T4=int4 -D DT=i32 -D USE_BLOCK_INDEX\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_with_block_index_u32.bin --options=\"-D T=uint -D T2=uint2 -D T3=uint3 -D T4=uint4 -D DT=u32 -D USE_BLOCK_INDEX\"
            fi
        fi
    done



