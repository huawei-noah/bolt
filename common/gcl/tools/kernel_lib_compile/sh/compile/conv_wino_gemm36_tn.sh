for file in *
    do
        if [ "${file##*.}"x = "cl"x ];then
             if [[ "${file}" == conv_wino_gemm36_tn.cl ]];then
                echo ./gcl_binary --input=$file --output=${file%.*}_13.bin  --options=\"${copt} -D LM=1 -D LN=3\"
                echo ./gcl_binary --input=$file --output=${file%.*}_14.bin  --options=\"${copt} -D LM=1 -D LN=4\"
                echo ./gcl_binary --input=$file --output=${file%.*}_15.bin  --options=\"${copt} -D LM=1 -D LN=5\"
                echo ./gcl_binary --input=$file --output=${file%.*}_16.bin  --options=\"${copt} -D LM=1 -D LN=6\"
                echo ./gcl_binary --input=$file --output=${file%.*}_17.bin  --options=\"${copt} -D LM=1 -D LN=7\"
                echo ./gcl_binary --input=$file --output=${file%.*}_18.bin  --options=\"${copt} -D LM=1 -D LN=8\"

                echo ./gcl_binary --input=$file --output=${file%.*}_22.bin  --options=\"${copt} -D LM=2 -D LN=2\"
                echo ./gcl_binary --input=$file --output=${file%.*}_23.bin  --options=\"${copt} -D LM=2 -D LN=3\"
                echo ./gcl_binary --input=$file --output=${file%.*}_24.bin  --options=\"${copt} -D LM=2 -D LN=4\"
                echo ./gcl_binary --input=$file --output=${file%.*}_25.bin  --options=\"${copt} -D LM=2 -D LN=5\"
                echo ./gcl_binary --input=$file --output=${file%.*}_26.bin  --options=\"${copt} -D LM=2 -D LN=6\"
                echo ./gcl_binary --input=$file --output=${file%.*}_27.bin  --options=\"${copt} -D LM=2 -D LN=7\"
                echo ./gcl_binary --input=$file --output=${file%.*}_28.bin  --options=\"${copt} -D LM=2 -D LN=8\"

                echo ./gcl_binary --input=$file --output=${file%.*}_31.bin  --options=\"${copt} -D LM=3 -D LN=1\"
                echo ./gcl_binary --input=$file --output=${file%.*}_32.bin  --options=\"${copt} -D LM=3 -D LN=2\"
                echo ./gcl_binary --input=$file --output=${file%.*}_33.bin  --options=\"${copt} -D LM=3 -D LN=3\"
                echo ./gcl_binary --input=$file --output=${file%.*}_34.bin  --options=\"${copt} -D LM=3 -D LN=4\"
                echo ./gcl_binary --input=$file --output=${file%.*}_35.bin  --options=\"${copt} -D LM=3 -D LN=5\"
                echo ./gcl_binary --input=$file --output=${file%.*}_36.bin  --options=\"${copt} -D LM=3 -D LN=6\"
                echo ./gcl_binary --input=$file --output=${file%.*}_37.bin  --options=\"${copt} -D LM=3 -D LN=7\"
                echo ./gcl_binary --input=$file --output=${file%.*}_38.bin  --options=\"${copt} -D LM=3 -D LN=8\"
                
                echo ./gcl_binary --input=$file --output=${file%.*}_41.bin  --options=\"${copt} -D LM=4 -D LN=1\"
                echo ./gcl_binary --input=$file --output=${file%.*}_42.bin  --options=\"${copt} -D LM=4 -D LN=2\"
                echo ./gcl_binary --input=$file --output=${file%.*}_43.bin  --options=\"${copt} -D LM=4 -D LN=3\"
                echo ./gcl_binary --input=$file --output=${file%.*}_44.bin  --options=\"${copt} -D LM=4 -D LN=4\"
                echo ./gcl_binary --input=$file --output=${file%.*}_45.bin  --options=\"${copt} -D LM=4 -D LN=5\"
                echo ./gcl_binary --input=$file --output=${file%.*}_46.bin  --options=\"${copt} -D LM=4 -D LN=6\"
                echo ./gcl_binary --input=$file --output=${file%.*}_47.bin  --options=\"${copt} -D LM=4 -D LN=7\"
                echo ./gcl_binary --input=$file --output=${file%.*}_48.bin  --options=\"${copt} -D LM=4 -D LN=8\"

                echo ./gcl_binary --input=$file --output=${file%.*}_51.bin  --options=\"${copt} -D LM=5 -D LN=1\"
                echo ./gcl_binary --input=$file --output=${file%.*}_52.bin  --options=\"${copt} -D LM=5 -D LN=2\"
                echo ./gcl_binary --input=$file --output=${file%.*}_53.bin  --options=\"${copt} -D LM=5 -D LN=3\"
                echo ./gcl_binary --input=$file --output=${file%.*}_54.bin  --options=\"${copt} -D LM=5 -D LN=4\"
                echo ./gcl_binary --input=$file --output=${file%.*}_55.bin  --options=\"${copt} -D LM=5 -D LN=5\"
                echo ./gcl_binary --input=$file --output=${file%.*}_56.bin  --options=\"${copt} -D LM=5 -D LN=6\"
                echo ./gcl_binary --input=$file --output=${file%.*}_57.bin  --options=\"${copt} -D LM=5 -D LN=7\"
                echo ./gcl_binary --input=$file --output=${file%.*}_58.bin  --options=\"${copt} -D LM=5 -D LN=8\"

                echo ./gcl_binary --input=$file --output=${file%.*}_61.bin  --options=\"${copt} -D LM=6 -D LN=1\"
                echo ./gcl_binary --input=$file --output=${file%.*}_62.bin  --options=\"${copt} -D LM=6 -D LN=2\"
                echo ./gcl_binary --input=$file --output=${file%.*}_63.bin  --options=\"${copt} -D LM=6 -D LN=3\"
                echo ./gcl_binary --input=$file --output=${file%.*}_64.bin  --options=\"${copt} -D LM=6 -D LN=4\"
                echo ./gcl_binary --input=$file --output=${file%.*}_65.bin  --options=\"${copt} -D LM=6 -D LN=5\"
                echo ./gcl_binary --input=$file --output=${file%.*}_66.bin  --options=\"${copt} -D LM=6 -D LN=6\"
                echo ./gcl_binary --input=$file --output=${file%.*}_67.bin  --options=\"${copt} -D LM=6 -D LN=7\"
                echo ./gcl_binary --input=$file --output=${file%.*}_68.bin  --options=\"${copt} -D LM=6 -D LN=8\"

                echo ./gcl_binary --input=$file --output=${file%.*}_71.bin  --options=\"${copt} -D LM=7 -D LN=1\"
                echo ./gcl_binary --input=$file --output=${file%.*}_72.bin  --options=\"${copt} -D LM=7 -D LN=2\"
                echo ./gcl_binary --input=$file --output=${file%.*}_73.bin  --options=\"${copt} -D LM=7 -D LN=3\"
                echo ./gcl_binary --input=$file --output=${file%.*}_74.bin  --options=\"${copt} -D LM=7 -D LN=4\"
                echo ./gcl_binary --input=$file --output=${file%.*}_75.bin  --options=\"${copt} -D LM=7 -D LN=5\"
                echo ./gcl_binary --input=$file --output=${file%.*}_76.bin  --options=\"${copt} -D LM=7 -D LN=6\"
                echo ./gcl_binary --input=$file --output=${file%.*}_77.bin  --options=\"${copt} -D LM=7 -D LN=7\"
                echo ./gcl_binary --input=$file --output=${file%.*}_78.bin  --options=\"${copt} -D LM=7 -D LN=8\"

                echo ./gcl_binary --input=$file --output=${file%.*}_81.bin  --options=\"${copt} -D LM=8 -D LN=1\"
                echo ./gcl_binary --input=$file --output=${file%.*}_82.bin  --options=\"${copt} -D LM=8 -D LN=2\"
                echo ./gcl_binary --input=$file --output=${file%.*}_83.bin  --options=\"${copt} -D LM=8 -D LN=3\"
                echo ./gcl_binary --input=$file --output=${file%.*}_84.bin  --options=\"${copt} -D LM=8 -D LN=4\"
                echo ./gcl_binary --input=$file --output=${file%.*}_85.bin  --options=\"${copt} -D LM=8 -D LN=5\"
                echo ./gcl_binary --input=$file --output=${file%.*}_86.bin  --options=\"${copt} -D LM=8 -D LN=6\"
                echo ./gcl_binary --input=$file --output=${file%.*}_87.bin  --options=\"${copt} -D LM=8 -D LN=7\"
                echo ./gcl_binary --input=$file --output=${file%.*}_88.bin  --options=\"${copt} -D LM=8 -D LN=8\"
             fi
        fi
    done



