for file in *
    do
        if [[ "${file}" == "bilateral_slice_apply_c12.cl" ]];then
            echo ./gcl_binary --input=$file --output=${file%.*}.bin  --options=\"${copt}\"
            echo ./gcl_binary --input=$file --output=${file%.*}_conv.bin  --options=\"${copt} -DCONV\"
            echo ./gcl_binary --input=$file --output=${file%.*}_uchar.bin  --options=\"${copt} -DUCHAR\"
            echo ./gcl_binary --input=$file --output=${file%.*}_conv_uchar.bin  --options=\"${copt} -DCONV -DUCHAR\"
        fi
    done



