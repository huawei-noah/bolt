for file in *
    do
        if [ "${file##*.}"x = "cl"x ];then
            clFileName=${file%.*}
            speConfig=0
            for filesh in *.sh
                do
                    if [ "${filesh##*.}"x = "sh"x ];then
                        shFileName=${filesh%.*}
                        if [ "$clFileName" = "$shFileName" ];then
                            speConfig=1;
                        fi
                    fi
                done
            if [ $speConfig -eq 0 ]; then
                echo ./gcl_binary --input=$file --output=${file%.*}.bin --options=\"${copt}\"
            fi
        fi
    done



