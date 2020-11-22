for file in *
    do
        if [ "${file##*.}"x = "cl"x ];then
            if [[ "${file}" == "deconv_gemm_trans_fltbuf.cl" ]];then
                echo ./gcl_binary --input=$file --output=${file%.*}_14.bin  --options=\"${copt}  -D C=1  -D K=4\"
                echo ./gcl_binary --input=$file --output=${file%.*}_24.bin  --options=\"${copt}  -D C=2  -D K=4\"
                echo ./gcl_binary --input=$file --output=${file%.*}_44.bin  --options=\"${copt}  -D C=4  -D K=4\"
            fi
        fi
    done



