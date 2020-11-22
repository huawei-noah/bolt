for file in *
    do
        if [ "${file##*.}"x = "cl"x ];then
            if [[ "${file}" == "conv_direct_trans_fltbuf.cl" ]];then
                echo ./gcl_binary --input=$file --output=${file%.*}_14.bin  --options=\"${copt}  -D C=1  -D K=4\"
                echo ./gcl_binary --input=$file --output=${file%.*}_41.bin  --options=\"${copt}  -D C=4  -D K=1\"
                echo ./gcl_binary --input=$file --output=${file%.*}_44.bin  --options=\"${copt}  -D C=4  -D K=4\"
                echo ./gcl_binary --input=$file --output=${file%.*}_48.bin  --options=\"${copt}  -D C=4  -D K=8\"
                echo ./gcl_binary --input=$file --output=${file%.*}_416.bin  --options=\"${copt} -D C=4  -D K=16\"
                echo ./gcl_binary --input=$file --output=${file%.*}_10.bin  --options=\"${copt}  -D C=1  -D K=0\"
                echo ./gcl_binary --input=$file --output=${file%.*}_20.bin  --options=\"${copt}  -D C=2  -D K=0\"
                echo ./gcl_binary --input=$file --output=${file%.*}_30.bin  --options=\"${copt}  -D C=3  -D K=0\"
                echo ./gcl_binary --input=$file --output=${file%.*}_40.bin  --options=\"${copt}  -D C=4  -D K=0\"
                echo ./gcl_binary --input=$file --output=${file%.*}_80.bin  --options=\"${copt}  -D C=8  -D K=0\"
                echo ./gcl_binary --input=$file --output=${file%.*}_160.bin  --options=\"${copt} -D C=16 -D K=0 \"
            fi
        fi
    done



