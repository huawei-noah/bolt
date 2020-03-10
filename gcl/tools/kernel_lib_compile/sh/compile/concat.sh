for file in *
    do
        if [ "${file##*.}"x = "cl"x ];then
            if [[ "${file}" == "concat.cl" ]];then
                 echo ./gcl_binary --input=$file --output=${file%.*}_11.bin --options=\"${copt} -D A=1 -D N=1\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_12.bin --options=\"${copt} -D A=1 -D N=2\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_13.bin --options=\"${copt} -D A=1 -D N=3\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_14.bin --options=\"${copt} -D A=1 -D N=4\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_15.bin --options=\"${copt} -D A=1 -D N=5\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_16.bin --options=\"${copt} -D A=1 -D N=6\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_17.bin --options=\"${copt} -D A=1 -D N=7\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_18.bin --options=\"${copt} -D A=1 -D N=8\"
            fi
        fi
    done



