for file in *
    do
        if [ "${file##*.}"x = "cl"x ];then
            if [[ "${file}" == "concat.cl" ]];then
                 echo ./gcl_binary --input=$file --output=${file%.*}_w1.bin --options=\"${copt} -D N=1 -D AXIS_W\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_w2.bin --options=\"${copt} -D N=2 -D AXIS_W\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_w3.bin --options=\"${copt} -D N=3 -D AXIS_W\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_w4.bin --options=\"${copt} -D N=4 -D AXIS_W\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_h1.bin --options=\"${copt} -D N=1 -D AXIS_H\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_h2.bin --options=\"${copt} -D N=2 -D AXIS_H\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_h3.bin --options=\"${copt} -D N=3 -D AXIS_H\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_h4.bin --options=\"${copt} -D N=4 -D AXIS_H\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_c1.bin --options=\"${copt} -D N=1 -D AXIS_C\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_c2.bin --options=\"${copt} -D N=2 -D AXIS_C\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_c3.bin --options=\"${copt} -D N=3 -D AXIS_C\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_c4.bin --options=\"${copt} -D N=4 -D AXIS_C\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_nonalign_c_p1_1.bin --options=\"${copt} -D N=1 -D AXIS_C -D NON_ALIGN_C\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_nonalign_c_p1_2.bin --options=\"${copt} -D N=2 -D AXIS_C -D NON_ALIGN_C\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_nonalign_c_p1_3.bin --options=\"${copt} -D N=3 -D AXIS_C -D NON_ALIGN_C\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_nonalign_c_p1_4.bin --options=\"${copt} -D N=4 -D AXIS_C -D NON_ALIGN_C\"
            fi
        fi
    done



