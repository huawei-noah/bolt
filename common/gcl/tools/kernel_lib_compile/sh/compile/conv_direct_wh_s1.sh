for file in *
    do
        if [ "${file##*.}"x = "cl"x ];then
            if [[ "${file}" == "conv_direct_wh_s1.cl" ]];then
                # W=4 H=1
                echo ./gcl_binary --input=$file --output=${file%.*}_4111.bin --options=\"${copt} -D W=4 -D H=1 -D ON=1 -D IN=4 -D LN=4 -D UN=3 -D Fsq=4 -D KN=1 -DUSE_HALF\"
                echo ./gcl_binary --input=$file --output=${file%.*}_4121.bin --options=\"${copt} -D W=4 -D H=1 -D ON=2 -D IN=5 -D LN=5 -D UN=4 -D Fsq=4 -D KN=1 -DUSE_HALF\"
                echo ./gcl_binary --input=$file --output=${file%.*}_4131.bin --options=\"${copt} -D W=4 -D H=1 -D ON=3 -D IN=6 -D LN=6 -D UN=5 -D Fsq=4 -D KN=1 -DUSE_HALF\"
                echo ./gcl_binary --input=$file --output=${file%.*}_4141.bin --options=\"${copt} -D W=4 -D H=1 -D ON=4 -D IN=6 -D LN=5 -D UN=5 -D Fsq=4 -D KN=1 -DUSE_HALF -D BASICE_REG\"
                echo ./gcl_binary --input=$file --output=${file%.*}_4151.bin --options=\"${copt} -D W=4 -D H=1 -D ON=5 -D IN=6 -D LN=5 -D UN=5 -D Fsq=4 -D KN=1 -DUSE_HALF -D BASICE_REG\"
                echo ./gcl_binary --input=$file --output=${file%.*}_4161.bin --options=\"${copt} -D W=4 -D H=1 -D ON=6 -D IN=6 -D LN=5 -D UN=5 -D Fsq=4 -D KN=1 -DUSE_HALF -D BASICE_REG\"
                echo ./gcl_binary --input=$file --output=${file%.*}_4171.bin --options=\"${copt} -D W=4 -D H=1 -D ON=7 -D IN=7 -D LN=6 -D UN=6 -D Fsq=4 -D KN=1 -DUSE_HALF -D BASICE_REG\"
                echo ./gcl_binary --input=$file --output=${file%.*}_4181.bin --options=\"${copt} -D W=4 -D H=1 -D ON=8 -D IN=8 -D LN=7 -D UN=7 -D Fsq=4 -D KN=1 -DUSE_HALF -D BASICE_REG\"
            
                echo ./gcl_binary --input=$file --output=${file%.*}_relu_4111.bin --options=\"${copt} -D W=4 -D H=1 -D ON=1 -D IN=4 -D LN=4 -D UN=3 -D Fsq=4 -D KN=1 -D USE_RELU -DUSE_HALF\"
                echo ./gcl_binary --input=$file --output=${file%.*}_relu_4121.bin --options=\"${copt} -D W=4 -D H=1 -D ON=2 -D IN=5 -D LN=5 -D UN=4 -D Fsq=4 -D KN=1 -D USE_RELU -DUSE_HALF\"
                echo ./gcl_binary --input=$file --output=${file%.*}_relu_4131.bin --options=\"${copt} -D W=4 -D H=1 -D ON=3 -D IN=6 -D LN=6 -D UN=5 -D Fsq=4 -D KN=1 -D USE_RELU -DUSE_HALF\"
                echo ./gcl_binary --input=$file --output=${file%.*}_relu_4141.bin --options=\"${copt} -D W=4 -D H=1 -D ON=4 -D IN=6 -D LN=5 -D UN=5 -D Fsq=4 -D KN=1 -D USE_RELU -DUSE_HALF -D BASICE_REG\"
                echo ./gcl_binary --input=$file --output=${file%.*}_relu_4151.bin --options=\"${copt} -D W=4 -D H=1 -D ON=5 -D IN=6 -D LN=5 -D UN=5 -D Fsq=4 -D KN=1 -D USE_RELU -DUSE_HALF -D BASICE_REG\"
                echo ./gcl_binary --input=$file --output=${file%.*}_relu_4161.bin --options=\"${copt} -D W=4 -D H=1 -D ON=6 -D IN=6 -D LN=5 -D UN=5 -D Fsq=4 -D KN=1 -D USE_RELU -DUSE_HALF -D BASICE_REG\"
                echo ./gcl_binary --input=$file --output=${file%.*}_relu_4171.bin --options=\"${copt} -D W=4 -D H=1 -D ON=7 -D IN=7 -D LN=6 -D UN=6 -D Fsq=4 -D KN=1 -D USE_RELU -DUSE_HALF -D BASICE_REG\"
                echo ./gcl_binary --input=$file --output=${file%.*}_relu_4181.bin --options=\"${copt} -D W=4 -D H=1 -D ON=8 -D IN=8 -D LN=7 -D UN=7 -D Fsq=4 -D KN=1 -D USE_RELU -DUSE_HALF -D BASICE_REG\"

                echo ./gcl_binary --input=$file --output=${file%.*}_relu6_4111.bin --options=\"${copt} -D W=4 -D H=1 -D ON=1 -D IN=4 -D LN=4 -D UN=3 -D Fsq=4 -D KN=1 -D USE_RELU6 -DUSE_HALF\"
                echo ./gcl_binary --input=$file --output=${file%.*}_relu6_4121.bin --options=\"${copt} -D W=4 -D H=1 -D ON=2 -D IN=5 -D LN=5 -D UN=4 -D Fsq=4 -D KN=1 -D USE_RELU6 -DUSE_HALF\"
                echo ./gcl_binary --input=$file --output=${file%.*}_relu6_4131.bin --options=\"${copt} -D W=4 -D H=1 -D ON=3 -D IN=6 -D LN=6 -D UN=5 -D Fsq=4 -D KN=1 -D USE_RELU6 -DUSE_HALF\"
                echo ./gcl_binary --input=$file --output=${file%.*}_relu6_4141.bin --options=\"${copt} -D W=4 -D H=1 -D ON=4 -D IN=6 -D LN=5 -D UN=5 -D Fsq=4 -D KN=1 -D USE_RELU6 -DUSE_HALF -D BASICE_REG\"
                echo ./gcl_binary --input=$file --output=${file%.*}_relu6_4151.bin --options=\"${copt} -D W=4 -D H=1 -D ON=5 -D IN=6 -D LN=5 -D UN=5 -D Fsq=4 -D KN=1 -D USE_RELU6 -DUSE_HALF -D BASICE_REG\"
                echo ./gcl_binary --input=$file --output=${file%.*}_relu6_4161.bin --options=\"${copt} -D W=4 -D H=1 -D ON=6 -D IN=6 -D LN=5 -D UN=5 -D Fsq=4 -D KN=1 -D USE_RELU6 -DUSE_HALF -D BASICE_REG\"
                echo ./gcl_binary --input=$file --output=${file%.*}_relu6_4171.bin --options=\"${copt} -D W=4 -D H=1 -D ON=7 -D IN=7 -D LN=6 -D UN=6 -D Fsq=4 -D KN=1 -D USE_RELU6 -DUSE_HALF -D BASICE_REG\"
                echo ./gcl_binary --input=$file --output=${file%.*}_relu6_4181.bin --options=\"${copt} -D W=4 -D H=1 -D ON=8 -D IN=8 -D LN=7 -D UN=7 -D Fsq=4 -D KN=1 -D USE_RELU6 -DUSE_HALF -D BASICE_REG\"

                echo ./gcl_binary --input=$file --output=${file%.*}_4112.bin --options=\"${copt} -D W=4 -D H=1 -D ON=1 -D IN=4 -D LN=4 -D UN=3 -D Fsq=4 -D KN=2 -DUSE_HALF\"
                echo ./gcl_binary --input=$file --output=${file%.*}_4122.bin --options=\"${copt} -D W=4 -D H=1 -D ON=2 -D IN=5 -D LN=5 -D UN=4 -D Fsq=4 -D KN=2 -DUSE_HALF\"
                echo ./gcl_binary --input=$file --output=${file%.*}_4132.bin --options=\"${copt} -D W=4 -D H=1 -D ON=3 -D IN=6 -D LN=6 -D UN=5 -D Fsq=4 -D KN=2 -DUSE_HALF\"
                echo ./gcl_binary --input=$file --output=${file%.*}_4142.bin --options=\"${copt} -D W=4 -D H=1 -D ON=4 -D IN=6 -D LN=5 -D UN=5 -D Fsq=4 -D KN=2 -DUSE_HALF -D BASICE_REG\"
            
                echo ./gcl_binary --input=$file --output=${file%.*}_relu_4112.bin --options=\"${copt} -D W=4 -D H=1 -D ON=1 -D IN=4 -D LN=4 -D UN=3 -D Fsq=4 -D KN=2 -DUSE_RELU -DUSE_HALF\"
                echo ./gcl_binary --input=$file --output=${file%.*}_relu_4122.bin --options=\"${copt} -D W=4 -D H=1 -D ON=2 -D IN=5 -D LN=5 -D UN=4 -D Fsq=4 -D KN=2 -DUSE_RELU -DUSE_HALF\"
                echo ./gcl_binary --input=$file --output=${file%.*}_relu_4132.bin --options=\"${copt} -D W=4 -D H=1 -D ON=3 -D IN=6 -D LN=6 -D UN=5 -D Fsq=4 -D KN=2 -DUSE_RELU -DUSE_HALF\"
                echo ./gcl_binary --input=$file --output=${file%.*}_relu_4142.bin --options=\"${copt} -D W=4 -D H=1 -D ON=4 -D IN=6 -D LN=5 -D UN=5 -D Fsq=4 -D KN=2 -DUSE_RELU -DUSE_HALF -D BASICE_REG\"

                echo ./gcl_binary --input=$file --output=${file%.*}_relu6_4112.bin --options=\"${copt} -D W=4 -D H=1 -D ON=1 -D IN=4 -D LN=4 -D UN=3 -D Fsq=4 -D KN=2 -DUSE_RELU6 -DUSE_HALF\"
                echo ./gcl_binary --input=$file --output=${file%.*}_relu6_4122.bin --options=\"${copt} -D W=4 -D H=1 -D ON=2 -D IN=5 -D LN=5 -D UN=4 -D Fsq=4 -D KN=2 -DUSE_RELU6 -DUSE_HALF\"
                echo ./gcl_binary --input=$file --output=${file%.*}_relu6_4132.bin --options=\"${copt} -D W=4 -D H=1 -D ON=3 -D IN=6 -D LN=6 -D UN=5 -D Fsq=4 -D KN=2 -DUSE_RELU6 -DUSE_HALF\"
                echo ./gcl_binary --input=$file --output=${file%.*}_relu6_4142.bin --options=\"${copt} -D W=4 -D H=1 -D ON=4 -D IN=6 -D LN=5 -D UN=5 -D Fsq=4 -D KN=2 -DUSE_RELU6 -DUSE_HALF -D BASICE_REG\"
                
                # W=3 H=1
                echo ./gcl_binary --input=$file --output=${file%.*}_3111.bin --options=\"${copt} -D W=3 -D H=1 -D ON=1 -D IN=3 -D LN=3 -D UN=2 -D Fsq=3 -D KN=1 -DUSE_HALF\"
                echo ./gcl_binary --input=$file --output=${file%.*}_3121.bin --options=\"${copt} -D W=3 -D H=1 -D ON=2 -D IN=4 -D LN=4 -D UN=3 -D Fsq=3 -D KN=1 -DUSE_HALF\"
                echo ./gcl_binary --input=$file --output=${file%.*}_3131.bin --options=\"${copt} -D W=3 -D H=1 -D ON=3 -D IN=5 -D LN=5 -D UN=4 -D Fsq=3 -D KN=1 -DUSE_HALF\"
                echo ./gcl_binary --input=$file --output=${file%.*}_3141.bin --options=\"${copt} -D W=3 -D H=1 -D ON=4 -D IN=4 -D LN=3 -D UN=3 -D Fsq=3 -D KN=1 -DUSE_HALF -D BASICE_REG\"
                echo ./gcl_binary --input=$file --output=${file%.*}_3151.bin --options=\"${copt} -D W=3 -D H=1 -D ON=5 -D IN=5 -D LN=4 -D UN=4 -D Fsq=3 -D KN=1 -DUSE_HALF -D BASICE_REG\"
                echo ./gcl_binary --input=$file --output=${file%.*}_3161.bin --options=\"${copt} -D W=3 -D H=1 -D ON=6 -D IN=6 -D LN=5 -D UN=5 -D Fsq=3 -D KN=1 -DUSE_HALF -D BASICE_REG\"
                echo ./gcl_binary --input=$file --output=${file%.*}_3171.bin --options=\"${copt} -D W=3 -D H=1 -D ON=7 -D IN=7 -D LN=6 -D UN=6 -D Fsq=3 -D KN=1 -DUSE_HALF -D BASICE_REG\"
                echo ./gcl_binary --input=$file --output=${file%.*}_3181.bin --options=\"${copt} -D W=3 -D H=1 -D ON=8 -D IN=8 -D LN=7 -D UN=7 -D Fsq=3 -D KN=1 -DUSE_HALF -D BASICE_REG\"
            
                echo ./gcl_binary --input=$file --output=${file%.*}_relu_3111.bin --options=\"${copt} -D W=3 -D H=1 -D ON=1 -D IN=3 -D LN=3 -D UN=2 -D Fsq=3 -D KN=1 -DUSE_RELU -DUSE_HALF\"
                echo ./gcl_binary --input=$file --output=${file%.*}_relu_3121.bin --options=\"${copt} -D W=3 -D H=1 -D ON=2 -D IN=4 -D LN=4 -D UN=3 -D Fsq=3 -D KN=1 -DUSE_RELU -DUSE_HALF\"
                echo ./gcl_binary --input=$file --output=${file%.*}_relu_3131.bin --options=\"${copt} -D W=3 -D H=1 -D ON=3 -D IN=5 -D LN=5 -D UN=4 -D Fsq=3 -D KN=1 -DUSE_RELU -DUSE_HALF\"
                echo ./gcl_binary --input=$file --output=${file%.*}_relu_3141.bin --options=\"${copt} -D W=3 -D H=1 -D ON=4 -D IN=4 -D LN=3 -D UN=3 -D Fsq=3 -D KN=1 -DUSE_RELU -DUSE_HALF -D BASICE_REG\"
                echo ./gcl_binary --input=$file --output=${file%.*}_relu_3151.bin --options=\"${copt} -D W=3 -D H=1 -D ON=5 -D IN=5 -D LN=4 -D UN=4 -D Fsq=3 -D KN=1 -DUSE_RELU -DUSE_HALF -D BASICE_REG\"
                echo ./gcl_binary --input=$file --output=${file%.*}_relu_3161.bin --options=\"${copt} -D W=3 -D H=1 -D ON=6 -D IN=6 -D LN=5 -D UN=5 -D Fsq=3 -D KN=1 -DUSE_RELU -DUSE_HALF -D BASICE_REG\"
                echo ./gcl_binary --input=$file --output=${file%.*}_relu_3171.bin --options=\"${copt} -D W=3 -D H=1 -D ON=7 -D IN=7 -D LN=6 -D UN=6 -D Fsq=3 -D KN=1 -DUSE_RELU -DUSE_HALF -D BASICE_REG\"
                echo ./gcl_binary --input=$file --output=${file%.*}_relu_3181.bin --options=\"${copt} -D W=3 -D H=1 -D ON=8 -D IN=8 -D LN=7 -D UN=7 -D Fsq=3 -D KN=1 -DUSE_RELU -DUSE_HALF -D BASICE_REG\"
                
                echo ./gcl_binary --input=$file --output=${file%.*}_relu6_3111.bin --options=\"${copt} -D W=3 -D H=1 -D ON=1 -D IN=3 -D LN=3 -D UN=2 -D Fsq=3 -D KN=1 -DUSE_RELU6 -DUSE_HALF\"
                echo ./gcl_binary --input=$file --output=${file%.*}_relu6_3121.bin --options=\"${copt} -D W=3 -D H=1 -D ON=2 -D IN=4 -D LN=4 -D UN=3 -D Fsq=3 -D KN=1 -DUSE_RELU6 -DUSE_HALF\"
                echo ./gcl_binary --input=$file --output=${file%.*}_relu6_3131.bin --options=\"${copt} -D W=3 -D H=1 -D ON=3 -D IN=5 -D LN=5 -D UN=4 -D Fsq=3 -D KN=1 -DUSE_RELU6 -DUSE_HALF\"
                echo ./gcl_binary --input=$file --output=${file%.*}_relu6_3141.bin --options=\"${copt} -D W=3 -D H=1 -D ON=4 -D IN=4 -D LN=3 -D UN=3 -D Fsq=3 -D KN=1 -DUSE_RELU6 -DUSE_HALF -D BASICE_REG\"
                echo ./gcl_binary --input=$file --output=${file%.*}_relu6_3151.bin --options=\"${copt} -D W=3 -D H=1 -D ON=5 -D IN=5 -D LN=4 -D UN=4 -D Fsq=3 -D KN=1 -DUSE_RELU6 -DUSE_HALF -D BASICE_REG\"
                echo ./gcl_binary --input=$file --output=${file%.*}_relu6_3161.bin --options=\"${copt} -D W=3 -D H=1 -D ON=6 -D IN=6 -D LN=5 -D UN=5 -D Fsq=3 -D KN=1 -DUSE_RELU6 -DUSE_HALF -D BASICE_REG\"
                echo ./gcl_binary --input=$file --output=${file%.*}_relu6_3171.bin --options=\"${copt} -D W=3 -D H=1 -D ON=7 -D IN=7 -D LN=6 -D UN=6 -D Fsq=3 -D KN=1 -DUSE_RELU6 -DUSE_HALF -D BASICE_REG\"
                echo ./gcl_binary --input=$file --output=${file%.*}_relu6_3181.bin --options=\"${copt} -D W=3 -D H=1 -D ON=8 -D IN=8 -D LN=7 -D UN=7 -D Fsq=3 -D KN=1 -DUSE_RELU6 -DUSE_HALF -D BASICE_REG\"


                echo ./gcl_binary --input=$file --output=${file%.*}_3112.bin --options=\"${copt} -D W=3 -D H=1 -D ON=1 -D IN=3 -D LN=3 -D UN=2 -D Fsq=3 -D KN=2 -DUSE_HALF\"
                echo ./gcl_binary --input=$file --output=${file%.*}_3122.bin --options=\"${copt} -D W=3 -D H=1 -D ON=2 -D IN=4 -D LN=4 -D UN=3 -D Fsq=3 -D KN=2 -DUSE_HALF\"
                echo ./gcl_binary --input=$file --output=${file%.*}_3132.bin --options=\"${copt} -D W=3 -D H=1 -D ON=3 -D IN=5 -D LN=5 -D UN=4 -D Fsq=3 -D KN=2 -DUSE_HALF\"
                echo ./gcl_binary --input=$file --output=${file%.*}_3142.bin --options=\"${copt} -D W=3 -D H=1 -D ON=4 -D IN=4 -D LN=3 -D UN=3 -D Fsq=3 -D KN=2 -DUSE_HALF -D BASICE_REG\"
            
                echo ./gcl_binary --input=$file --output=${file%.*}_relu_3112.bin --options=\"${copt} -D W=3 -D H=1 -D ON=1 -D IN=3 -D LN=3 -D UN=2 -D Fsq=3 -D KN=2 -DUSE_RELU -DUSE_HALF\"
                echo ./gcl_binary --input=$file --output=${file%.*}_relu_3122.bin --options=\"${copt} -D W=3 -D H=1 -D ON=2 -D IN=4 -D LN=4 -D UN=3 -D Fsq=3 -D KN=2 -DUSE_RELU -DUSE_HALF\"
                echo ./gcl_binary --input=$file --output=${file%.*}_relu_3132.bin --options=\"${copt} -D W=3 -D H=1 -D ON=3 -D IN=5 -D LN=5 -D UN=4 -D Fsq=3 -D KN=2 -DUSE_RELU -DUSE_HALF\"
                echo ./gcl_binary --input=$file --output=${file%.*}_relu_3142.bin --options=\"${copt} -D W=3 -D H=1 -D ON=4 -D IN=4 -D LN=3 -D UN=3 -D Fsq=3 -D KN=2 -DUSE_RELU -DUSE_HALF -D BASICE_REG\"
                
                echo ./gcl_binary --input=$file --output=${file%.*}_relu6_3112.bin --options=\"${copt} -D W=3 -D H=1 -D ON=1 -D IN=3 -D LN=3 -D UN=2 -D Fsq=3 -D KN=2 -DUSE_RELU6 -DUSE_HALF\"
                echo ./gcl_binary --input=$file --output=${file%.*}_relu6_3122.bin --options=\"${copt} -D W=3 -D H=1 -D ON=2 -D IN=4 -D LN=4 -D UN=3 -D Fsq=3 -D KN=2 -DUSE_RELU6 -DUSE_HALF\"
                echo ./gcl_binary --input=$file --output=${file%.*}_relu6_3132.bin --options=\"${copt} -D W=3 -D H=1 -D ON=3 -D IN=5 -D LN=5 -D UN=4 -D Fsq=3 -D KN=2 -DUSE_RELU6 -DUSE_HALF\"
                echo ./gcl_binary --input=$file --output=${file%.*}_relu6_3142.bin --options=\"${copt} -D W=3 -D H=1 -D ON=4 -D IN=4 -D LN=3 -D UN=3 -D Fsq=3 -D KN=2 -DUSE_RELU6 -DUSE_HALF -D BASICE_REG\"

                # W=1 H=4
                echo ./gcl_binary --input=$file --output=${file%.*}_1411.bin --options=\"${copt} -D W=1 -D H=4 -D ON=1 -D IN=1 -D LN=1 -D Fsq=4 -D KN=1 -DUSE_HALF\"
                echo ./gcl_binary --input=$file --output=${file%.*}_1421.bin --options=\"${copt} -D W=1 -D H=4 -D ON=2 -D IN=2 -D LN=2 -D Fsq=4 -D KN=1 -DUSE_HALF\"
                echo ./gcl_binary --input=$file --output=${file%.*}_1431.bin --options=\"${copt} -D W=1 -D H=4 -D ON=3 -D IN=3 -D LN=3 -D Fsq=4 -D KN=1 -DUSE_HALF\"
                echo ./gcl_binary --input=$file --output=${file%.*}_1441.bin --options=\"${copt} -D W=1 -D H=4 -D ON=4 -D IN=4 -D LN=4 -D Fsq=4 -D KN=1 -DUSE_HALF\"
                echo ./gcl_binary --input=$file --output=${file%.*}_1451.bin --options=\"${copt} -D W=1 -D H=4 -D ON=5 -D IN=5 -D LN=5 -D Fsq=4 -D KN=1 -DUSE_HALF\"
                echo ./gcl_binary --input=$file --output=${file%.*}_1461.bin --options=\"${copt} -D W=1 -D H=4 -D ON=6 -D IN=6 -D LN=6 -D Fsq=4 -D KN=1 -DUSE_HALF\"
                echo ./gcl_binary --input=$file --output=${file%.*}_1471.bin --options=\"${copt} -D W=1 -D H=4 -D ON=7 -D IN=7 -D LN=7 -D Fsq=4 -D KN=1 -DUSE_HALF\"
                echo ./gcl_binary --input=$file --output=${file%.*}_1481.bin --options=\"${copt} -D W=1 -D H=4 -D ON=8 -D IN=8 -D LN=8 -D Fsq=4 -D KN=1 -DUSE_HALF\"

 
                echo ./gcl_binary --input=$file --output=${file%.*}_relu_1411.bin --options=\"${copt} -D W=1 -D H=4 -D ON=1 -D IN=1 -D LN=1 -D Fsq=4 -D KN=1 -DUSE_RELU -DUSE_HALF\"
                echo ./gcl_binary --input=$file --output=${file%.*}_relu_1421.bin --options=\"${copt} -D W=1 -D H=4 -D ON=2 -D IN=2 -D LN=2 -D Fsq=4 -D KN=1 -DUSE_RELU -DUSE_HALF\"
                echo ./gcl_binary --input=$file --output=${file%.*}_relu_1431.bin --options=\"${copt} -D W=1 -D H=4 -D ON=3 -D IN=3 -D LN=3 -D Fsq=4 -D KN=1 -DUSE_RELU -DUSE_HALF\"
                echo ./gcl_binary --input=$file --output=${file%.*}_relu_1441.bin --options=\"${copt} -D W=1 -D H=4 -D ON=4 -D IN=4 -D LN=4 -D Fsq=4 -D KN=1 -DUSE_RELU -DUSE_HALF\"
                echo ./gcl_binary --input=$file --output=${file%.*}_relu_1451.bin --options=\"${copt} -D W=1 -D H=4 -D ON=5 -D IN=5 -D LN=5 -D Fsq=4 -D KN=1 -DUSE_RELU -DUSE_HALF\"
                echo ./gcl_binary --input=$file --output=${file%.*}_relu_1461.bin --options=\"${copt} -D W=1 -D H=4 -D ON=6 -D IN=6 -D LN=6 -D Fsq=4 -D KN=1 -DUSE_RELU -DUSE_HALF\"
                echo ./gcl_binary --input=$file --output=${file%.*}_relu_1471.bin --options=\"${copt} -D W=1 -D H=4 -D ON=7 -D IN=7 -D LN=7 -D Fsq=4 -D KN=1 -DUSE_RELU -DUSE_HALF\"
                echo ./gcl_binary --input=$file --output=${file%.*}_relu_1481.bin --options=\"${copt} -D W=1 -D H=4 -D ON=8 -D IN=8 -D LN=8 -D Fsq=4 -D KN=1 -DUSE_RELU -DUSE_HALF\"

                echo ./gcl_binary --input=$file --output=${file%.*}_relu6_1411.bin --options=\"${copt} -D W=1 -D H=4 -D ON=1 -D IN=1 -D LN=1 -D Fsq=4 -D KN=1 -DUSE_RELU6 -DUSE_HALF\"
                echo ./gcl_binary --input=$file --output=${file%.*}_relu6_1421.bin --options=\"${copt} -D W=1 -D H=4 -D ON=2 -D IN=2 -D LN=2 -D Fsq=4 -D KN=1 -DUSE_RELU6 -DUSE_HALF\"
                echo ./gcl_binary --input=$file --output=${file%.*}_relu6_1431.bin --options=\"${copt} -D W=1 -D H=4 -D ON=3 -D IN=3 -D LN=3 -D Fsq=4 -D KN=1 -DUSE_RELU6 -DUSE_HALF\"
                echo ./gcl_binary --input=$file --output=${file%.*}_relu6_1441.bin --options=\"${copt} -D W=1 -D H=4 -D ON=4 -D IN=4 -D LN=4 -D Fsq=4 -D KN=1 -DUSE_RELU6 -DUSE_HALF\"
                echo ./gcl_binary --input=$file --output=${file%.*}_relu6_1451.bin --options=\"${copt} -D W=1 -D H=4 -D ON=5 -D IN=5 -D LN=5 -D Fsq=4 -D KN=1 -DUSE_RELU6 -DUSE_HALF\"
                echo ./gcl_binary --input=$file --output=${file%.*}_relu6_1461.bin --options=\"${copt} -D W=1 -D H=4 -D ON=6 -D IN=6 -D LN=6 -D Fsq=4 -D KN=1 -DUSE_RELU6 -DUSE_HALF\"
                echo ./gcl_binary --input=$file --output=${file%.*}_relu6_1471.bin --options=\"${copt} -D W=1 -D H=4 -D ON=7 -D IN=7 -D LN=7 -D Fsq=4 -D KN=1 -DUSE_RELU6 -DUSE_HALF\"
                echo ./gcl_binary --input=$file --output=${file%.*}_relu6_1481.bin --options=\"${copt} -D W=1 -D H=4 -D ON=8 -D IN=8 -D LN=8 -D Fsq=4 -D KN=1 -DUSE_RELU6 -DUSE_HALF\"

 
                echo ./gcl_binary --input=$file --output=${file%.*}_1412.bin --options=\"${copt} -D W=1 -D H=4 -D ON=1 -D IN=1 -D LN=1 -D Fsq=4 -D KN=2 -DUSE_HALF\"
                echo ./gcl_binary --input=$file --output=${file%.*}_1422.bin --options=\"${copt} -D W=1 -D H=4 -D ON=2 -D IN=2 -D LN=2 -D Fsq=4 -D KN=2 -DUSE_HALF\"
                echo ./gcl_binary --input=$file --output=${file%.*}_1432.bin --options=\"${copt} -D W=1 -D H=4 -D ON=3 -D IN=3 -D LN=3 -D Fsq=4 -D KN=2 -DUSE_HALF\"
                echo ./gcl_binary --input=$file --output=${file%.*}_1442.bin --options=\"${copt} -D W=1 -D H=4 -D ON=4 -D IN=4 -D LN=4 -D Fsq=4 -D KN=2 -DUSE_HALF\"
 
                echo ./gcl_binary --input=$file --output=${file%.*}_relu_1412.bin --options=\"${copt} -D W=1 -D H=4 -D ON=1 -D IN=1 -D LN=1 -D Fsq=4 -D KN=2 -DUSE_RELU -DUSE_HALF\"
                echo ./gcl_binary --input=$file --output=${file%.*}_relu_1422.bin --options=\"${copt} -D W=1 -D H=4 -D ON=2 -D IN=2 -D LN=2 -D Fsq=4 -D KN=2 -DUSE_RELU -DUSE_HALF\"
                echo ./gcl_binary --input=$file --output=${file%.*}_relu_1432.bin --options=\"${copt} -D W=1 -D H=4 -D ON=3 -D IN=3 -D LN=3 -D Fsq=4 -D KN=2 -DUSE_RELU -DUSE_HALF\"
                echo ./gcl_binary --input=$file --output=${file%.*}_relu_1442.bin --options=\"${copt} -D W=1 -D H=4 -D ON=4 -D IN=4 -D LN=4 -D Fsq=4 -D KN=2 -DUSE_RELU -DUSE_HALF\"

                echo ./gcl_binary --input=$file --output=${file%.*}_relu6_1412.bin --options=\"${copt} -D W=1 -D H=4 -D ON=1 -D IN=1 -D LN=1 -D Fsq=4 -D KN=2 -DUSE_RELU6 -DUSE_HALF\"
                echo ./gcl_binary --input=$file --output=${file%.*}_relu6_1422.bin --options=\"${copt} -D W=1 -D H=4 -D ON=2 -D IN=2 -D LN=2 -D Fsq=4 -D KN=2 -DUSE_RELU6 -DUSE_HALF\"
                echo ./gcl_binary --input=$file --output=${file%.*}_relu6_1432.bin --options=\"${copt} -D W=1 -D H=4 -D ON=3 -D IN=3 -D LN=3 -D Fsq=4 -D KN=2 -DUSE_RELU6 -DUSE_HALF\"
                echo ./gcl_binary --input=$file --output=${file%.*}_relu6_1442.bin --options=\"${copt} -D W=1 -D H=4 -D ON=4 -D IN=4 -D LN=4 -D Fsq=4 -D KN=2 -DUSE_RELU6 -DUSE_HALF\"
            fi
        fi
    done





