for file in *
    do
        if [ "${file##*.}"x = "cl"x ];then
            if [[ "${file}" == "eltwise_broadcast.cl" ]];then
                 echo ./gcl_binary --input=$file --output=${file%.*}_max_xyz.bin --options=\"${copt} -D TP=max_ -D B_AXIS=xyz -DBROAD_XYZ  -DUSE_MAX\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_max_xy.bin  --options=\"${copt} -D TP=max_ -D B_AXIS=xy  -DBROAD_XY   -DUSE_MAX\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_max_y.bin   --options=\"${copt} -D TP=max_ -D B_AXIS=y   -DBROAD_Y    -DUSE_MAX\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_max_x.bin   --options=\"${copt} -D TP=max_ -D B_AXIS=x   -DBROAD_X    -DUSE_MAX\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_sum_xyz.bin --options=\"${copt} -D TP=max_ -D B_AXIS=xyz -DBROAD_XYZ  -DUSE_SUM\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_sum_xy.bin  --options=\"${copt} -D TP=max_ -D B_AXIS=xy  -DBROAD_XY   -DUSE_SUM\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_sum_y.bin   --options=\"${copt} -D TP=max_ -D B_AXIS=y   -DBROAD_Y    -DUSE_SUM\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_sum_x.bin   --options=\"${copt} -D TP=max_ -D B_AXIS=x   -DBROAD_X    -DUSE_SUM\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_prod_xyz.bin --options=\"${copt} -D TP=max_ -D B_AXIS=xyz -DBROAD_XYZ -DUSE_PROD\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_prod_xy.bin  --options=\"${copt} -D TP=max_ -D B_AXIS=xy  -DBROAD_XY  -DUSE_PROD\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_prod_x.bin   --options=\"${copt} -D TP=max_ -D B_AXIS=x   -DBROAD_X   -DUSE_PROD\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_prod_y.bin   --options=\"${copt} -D TP=max_ -D B_AXIS=y   -DBROAD_Y   -DUSE_PROD\"

                 echo ./gcl_binary --input=$file --output=${file%.*}_nchw_max_xyz.bin --options=\"${copt} -D TP=max_ -D B_AXIS=xyz -DBROAD_XYZ -DUSE_NCHW -DUSE_MAX\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_nchw_max_xy.bin  --options=\"${copt} -D TP=max_ -D B_AXIS=xy  -DBROAD_XY  -DUSE_NCHW -DUSE_MAX\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_nchw_max_y.bin   --options=\"${copt} -D TP=max_ -D B_AXIS=y   -DBROAD_Y   -DUSE_NCHW -DUSE_MAX\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_nchw_max_x.bin   --options=\"${copt} -D TP=max_ -D B_AXIS=x   -DBROAD_X   -DUSE_NCHW -DUSE_MAX\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_nchw_sum_xyz.bin --options=\"${copt} -D TP=max_ -D B_AXIS=xyz -DBROAD_XYZ -DUSE_NCHW -DUSE_SUM\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_nchw_sum_xy.bin  --options=\"${copt} -D TP=max_ -D B_AXIS=xy  -DBROAD_XY  -DUSE_NCHW -DUSE_SUM\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_nchw_sum_y.bin   --options=\"${copt} -D TP=max_ -D B_AXIS=y   -DBROAD_Y   -DUSE_NCHW -DUSE_SUM\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_nchw_sum_x.bin   --options=\"${copt} -D TP=max_ -D B_AXIS=x   -DBROAD_X   -DUSE_NCHW -DUSE_SUM\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_nchw_prod_xyz.bin --options=\"${copt} -D TP=max_ -D B_AXIS=xyz -DBROAD_XYZ-DUSE_NCHW -DUSE_PROD\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_nchw_prod_xy.bin  --options=\"${copt} -D TP=max_ -D B_AXIS=xy  -DBROAD_XY -DUSE_NCHW -DUSE_PROD\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_nchw_prod_y.bin   --options=\"${copt} -D TP=max_ -D B_AXIS=y   -DBROAD_Y  -DUSE_NCHW -DUSE_PROD\"
                 echo ./gcl_binary --input=$file --output=${file%.*}_nchw_prod_x.bin   --options=\"${copt} -D TP=max_ -D B_AXIS=x   -DBROAD_X  -DUSE_NCHW -DUSE_PROD\"
            fi
        fi
    done

