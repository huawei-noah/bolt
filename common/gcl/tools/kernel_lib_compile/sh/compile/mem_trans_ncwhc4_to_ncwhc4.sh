for file in *
    do
        if [ "${file##*.}"x = "cl"x ];then
            if [[ "${file}" == "mem_trans_ncwhc4_to_ncwhc4.cl" ]];then
                 echo ./gcl_binary --input=$file --output=${file%.*}.bin --options=\"${copt}\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_output_tran.bin --options=\"${copt} -DOUTPUT_TRAN\"
            fi
        fi
    done



