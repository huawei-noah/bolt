for file in *
    do
        if [ "${file##*.}"x = "cl"x ];then
             if [[ "${file}" == "conv_direct_3d_sw1_nchw_to_ncwhc4.cl" ]];then
                echo ./gcl_binary --input=$file --output=${file%.*}_relu_752.bin  --options=\"${copt} -D FWH=7 -D FT=5 -D FWHT=245 -D ON=2 -DUSE_RELU\"
             fi
        fi
    done



