for file in *
    do
        if [ "${file##*.}"x = "cl"x ];then
            if [[ "${file}" == "activation.cl" ]];then
                 echo ./gcl_binary --input=$file --output=${file%.*}_relu1.bin     --options=\"${copt} -D AC=relu     -D H=1 -DUSE_RELU\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_relu61.bin    --options=\"${copt} -D AC=relu6    -D H=1 -DUSE_RELU6\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_hsigmoid1.bin --options=\"${copt} -D AC=hsigmoid -D H=1 -DUSE_HSIGMOID\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_hswish1.bin   --options=\"${copt} -D AC=hswish   -D H=1 -DUSE_HSWISH\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_gelu1.bin     --options=\"${copt} -D AC=gelu     -D H=1 -DUSE_GELU\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_tanh1.bin     --options=\"${copt} -D AC=tanh     -D H=1 -DUSE_TANH\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_sigmoid1.bin  --options=\"${copt} -D AC=sigmoid  -D H=1 -DUSE_SIGMOID\"
            fi
        fi
    done



