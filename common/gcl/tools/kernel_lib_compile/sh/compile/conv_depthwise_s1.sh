for file in *
    do
        if [ "${file##*.}"x = "cl"x ];then
            if [[ "${file}" == "conv_depthwise_s1.cl" ]];then
            echo ./gcl_binary --input=$file --output=${file%.*}_31.bin  --options=\"${copt} -D F=3 -D ON=1 -D IN=3 -D LN=3 -D UN=2 -D Fsq=9  -DUSE_HALF\"
            echo ./gcl_binary --input=$file --output=${file%.*}_32.bin  --options=\"${copt} -D F=3 -D ON=2 -D IN=4 -D LN=4 -D UN=3 -D Fsq=9  -DUSE_HALF\"
            echo ./gcl_binary --input=$file --output=${file%.*}_33.bin  --options=\"${copt} -D F=3 -D ON=3 -D IN=5 -D LN=5 -D UN=4 -D Fsq=9  -DUSE_HALF\"
            echo ./gcl_binary --input=$file --output=${file%.*}_34.bin  --options=\"${copt} -D F=3 -D ON=4 -D IN=6 -D LN=6 -D UN=5 -D Fsq=9  -DUSE_HALF\"
            echo ./gcl_binary --input=$file --output=${file%.*}_35.bin  --options=\"${copt} -D F=3 -D ON=5 -D IN=7 -D LN=7 -D UN=6 -D Fsq=9  -DUSE_HALF\"
            echo ./gcl_binary --input=$file --output=${file%.*}_36.bin  --options=\"${copt} -D F=3 -D ON=6 -D IN=6 -D LN=5 -D UN=5 -D Fsq=9  -DUSE_HALF -D BASICE_REG\"
            echo ./gcl_binary --input=$file --output=${file%.*}_37.bin  --options=\"${copt} -D F=3 -D ON=7 -D IN=7 -D LN=6 -D UN=6 -D Fsq=9  -DUSE_HALF -D BASICE_REG\"
            echo ./gcl_binary --input=$file --output=${file%.*}_38.bin  --options=\"${copt} -D F=3 -D ON=8 -D IN=8 -D LN=7 -D UN=7 -D Fsq=9  -DUSE_HALF -D BASICE_REG\"

            echo ./gcl_binary --input=$file --output=${file%.*}_relu_31.bin  --options=\"${copt} -D F=3 -D ON=1 -D IN=3 -D LN=3 -D UN=2 -D Fsq=9  -DUSE_HALF -DUSE_RELU\"
            echo ./gcl_binary --input=$file --output=${file%.*}_relu_32.bin  --options=\"${copt} -D F=3 -D ON=2 -D IN=4 -D LN=4 -D UN=3 -D Fsq=9  -DUSE_HALF -DUSE_RELU\"
            echo ./gcl_binary --input=$file --output=${file%.*}_relu_33.bin  --options=\"${copt} -D F=3 -D ON=3 -D IN=5 -D LN=5 -D UN=4 -D Fsq=9  -DUSE_HALF -DUSE_RELU\"
            echo ./gcl_binary --input=$file --output=${file%.*}_relu_34.bin  --options=\"${copt} -D F=3 -D ON=4 -D IN=6 -D LN=6 -D UN=5 -D Fsq=9  -DUSE_HALF -DUSE_RELU\"
            echo ./gcl_binary --input=$file --output=${file%.*}_relu_35.bin  --options=\"${copt} -D F=3 -D ON=5 -D IN=7 -D LN=7 -D UN=6 -D Fsq=9  -DUSE_HALF -DUSE_RELU\"
            echo ./gcl_binary --input=$file --output=${file%.*}_relu_36.bin  --options=\"${copt} -D F=3 -D ON=6 -D IN=6 -D LN=5 -D UN=5 -D Fsq=9  -DUSE_HALF -DUSE_RELU -D BASICE_REG\"
            echo ./gcl_binary --input=$file --output=${file%.*}_relu_37.bin  --options=\"${copt} -D F=3 -D ON=7 -D IN=7 -D LN=6 -D UN=6 -D Fsq=9  -DUSE_HALF -DUSE_RELU -D BASICE_REG\"
            echo ./gcl_binary --input=$file --output=${file%.*}_relu_38.bin  --options=\"${copt} -D F=3 -D ON=8 -D IN=8 -D LN=7 -D UN=7 -D Fsq=9  -DUSE_HALF -DUSE_RELU -D BASICE_REG\"


            echo ./gcl_binary --input=$file --output=${file%.*}_relu6_31.bin  --options=\"${copt} -D F=3 -D ON=1 -D IN=3 -D LN=3 -D UN=2 -D Fsq=9  -DUSE_HALF -DUSE_RELU6\"
            echo ./gcl_binary --input=$file --output=${file%.*}_relu6_32.bin  --options=\"${copt} -D F=3 -D ON=2 -D IN=4 -D LN=4 -D UN=3 -D Fsq=9  -DUSE_HALF -DUSE_RELU6\"
            echo ./gcl_binary --input=$file --output=${file%.*}_relu6_33.bin  --options=\"${copt} -D F=3 -D ON=3 -D IN=5 -D LN=5 -D UN=4 -D Fsq=9  -DUSE_HALF -DUSE_RELU6\"
            echo ./gcl_binary --input=$file --output=${file%.*}_relu6_34.bin  --options=\"${copt} -D F=3 -D ON=4 -D IN=6 -D LN=6 -D UN=5 -D Fsq=9  -DUSE_HALF -DUSE_RELU6\"
            echo ./gcl_binary --input=$file --output=${file%.*}_relu6_35.bin  --options=\"${copt} -D F=3 -D ON=5 -D IN=7 -D LN=7 -D UN=6 -D Fsq=9  -DUSE_HALF -DUSE_RELU6\"
            echo ./gcl_binary --input=$file --output=${file%.*}_relu6_36.bin  --options=\"${copt} -D F=3 -D ON=6 -D IN=6 -D LN=5 -D UN=5 -D Fsq=9  -DUSE_HALF -DUSE_RELU6 -D BASICE_REG\"
            echo ./gcl_binary --input=$file --output=${file%.*}_relu6_37.bin  --options=\"${copt} -D F=3 -D ON=7 -D IN=7 -D LN=6 -D UN=6 -D Fsq=9  -DUSE_HALF -DUSE_RELU6 -D BASICE_REG\"
            echo ./gcl_binary --input=$file --output=${file%.*}_relu6_38.bin  --options=\"${copt} -D F=3 -D ON=8 -D IN=8 -D LN=7 -D UN=7 -D Fsq=9  -DUSE_HALF -DUSE_RELU6 -D BASICE_REG\"


            echo ./gcl_binary --input=$file --output=${file%.*}_51.bin  --options=\"${copt} -D F=5 -D ON=1 -D IN=5 -D LN=5 -D UN=4 -D Fsq=25 -DUSE_HALF\"
            echo ./gcl_binary --input=$file --output=${file%.*}_52.bin  --options=\"${copt} -D F=5 -D ON=2 -D IN=6 -D LN=6 -D UN=5 -D Fsq=25 -DUSE_HALF\"
            echo ./gcl_binary --input=$file --output=${file%.*}_53.bin  --options=\"${copt} -D F=5 -D ON=3 -D IN=7 -D LN=7 -D UN=6 -D Fsq=25 -DUSE_HALF\"
            echo ./gcl_binary --input=$file --output=${file%.*}_54.bin  --options=\"${copt} -D F=5 -D ON=4 -D IN=4 -D LN=3 -D UN=3 -D Fsq=25 -DUSE_HALF -D BASICE_REG\"
            echo ./gcl_binary --input=$file --output=${file%.*}_55.bin  --options=\"${copt} -D F=5 -D ON=5 -D IN=5 -D LN=4 -D UN=4 -D Fsq=25 -DUSE_HALF -D BASICE_REG\"
            echo ./gcl_binary --input=$file --output=${file%.*}_56.bin  --options=\"${copt} -D F=5 -D ON=6 -D IN=6 -D LN=5 -D UN=5 -D Fsq=25 -DUSE_HALF -D BASICE_REG\"
            echo ./gcl_binary --input=$file --output=${file%.*}_57.bin  --options=\"${copt} -D F=5 -D ON=7 -D IN=7 -D LN=6 -D UN=6 -D Fsq=25 -DUSE_HALF -D BASICE_REG\"
            echo ./gcl_binary --input=$file --output=${file%.*}_58.bin  --options=\"${copt} -D F=5 -D ON=8 -D IN=8 -D LN=7 -D UN=7 -D Fsq=25 -DUSE_HALF -D BASICE_REG\"

            echo ./gcl_binary --input=$file --output=${file%.*}_relu_51.bin  --options=\"${copt} -D F=5 -D ON=1 -D IN=5 -D LN=5 -D UN=4 -D Fsq=25 -DUSE_HALF -DUSE_RELU\"
            echo ./gcl_binary --input=$file --output=${file%.*}_relu_52.bin  --options=\"${copt} -D F=5 -D ON=2 -D IN=6 -D LN=6 -D UN=5 -D Fsq=25 -DUSE_HALF -DUSE_RELU\"
            echo ./gcl_binary --input=$file --output=${file%.*}_relu_53.bin  --options=\"${copt} -D F=5 -D ON=3 -D IN=7 -D LN=7 -D UN=6 -D Fsq=25 -DUSE_HALF -DUSE_RELU\"
            echo ./gcl_binary --input=$file --output=${file%.*}_relu_54.bin  --options=\"${copt} -D F=5 -D ON=4 -D IN=4 -D LN=3 -D UN=3 -D Fsq=25 -DUSE_HALF -DUSE_RELU -D BASICE_REG\"
            echo ./gcl_binary --input=$file --output=${file%.*}_relu_55.bin  --options=\"${copt} -D F=5 -D ON=5 -D IN=5 -D LN=4 -D UN=4 -D Fsq=25 -DUSE_HALF -DUSE_RELU -D BASICE_REG\"
            echo ./gcl_binary --input=$file --output=${file%.*}_relu_56.bin  --options=\"${copt} -D F=5 -D ON=6 -D IN=6 -D LN=5 -D UN=5 -D Fsq=25 -DUSE_HALF -DUSE_RELU -D BASICE_REG\"
            echo ./gcl_binary --input=$file --output=${file%.*}_relu_57.bin  --options=\"${copt} -D F=5 -D ON=7 -D IN=7 -D LN=6 -D UN=6 -D Fsq=25 -DUSE_HALF -DUSE_RELU -D BASICE_REG\"
            echo ./gcl_binary --input=$file --output=${file%.*}_relu_58.bin  --options=\"${copt} -D F=5 -D ON=8 -D IN=8 -D LN=7 -D UN=7 -D Fsq=25 -DUSE_HALF -DUSE_RELU -D BASICE_REG\"


            echo ./gcl_binary --input=$file --output=${file%.*}_relu6_51.bin  --options=\"${copt} -D F=5 -D ON=1 -D IN=5 -D LN=5 -D UN=4 -D Fsq=25 -DUSE_HALF -DUSE_RELU6\"
            echo ./gcl_binary --input=$file --output=${file%.*}_relu6_52.bin  --options=\"${copt} -D F=5 -D ON=2 -D IN=6 -D LN=6 -D UN=5 -D Fsq=25 -DUSE_HALF -DUSE_RELU6\"
            echo ./gcl_binary --input=$file --output=${file%.*}_relu6_53.bin  --options=\"${copt} -D F=5 -D ON=3 -D IN=7 -D LN=7 -D UN=6 -D Fsq=25 -DUSE_HALF -DUSE_RELU6\"
            echo ./gcl_binary --input=$file --output=${file%.*}_relu6_54.bin  --options=\"${copt} -D F=5 -D ON=4 -D IN=4 -D LN=3 -D UN=3 -D Fsq=25 -DUSE_HALF -DUSE_RELU6 -D BASICE_REG\"
            echo ./gcl_binary --input=$file --output=${file%.*}_relu6_55.bin  --options=\"${copt} -D F=5 -D ON=5 -D IN=5 -D LN=4 -D UN=4 -D Fsq=25 -DUSE_HALF -DUSE_RELU6 -D BASICE_REG\"
            echo ./gcl_binary --input=$file --output=${file%.*}_relu6_56.bin  --options=\"${copt} -D F=5 -D ON=6 -D IN=6 -D LN=5 -D UN=5 -D Fsq=25 -DUSE_HALF -DUSE_RELU6 -D BASICE_REG\"
            echo ./gcl_binary --input=$file --output=${file%.*}_relu6_57.bin  --options=\"${copt} -D F=5 -D ON=7 -D IN=7 -D LN=6 -D UN=6 -D Fsq=25 -DUSE_HALF -DUSE_RELU6 -D BASICE_REG\"
            echo ./gcl_binary --input=$file --output=${file%.*}_relu6_58.bin  --options=\"${copt} -D F=5 -D ON=8 -D IN=8 -D LN=7 -D UN=7 -D Fsq=25 -DUSE_HALF -DUSE_RELU6 -D BASICE_REG\"

            echo ./gcl_binary --input=$file --output=${file%.*}_71.bin  --options=\"${copt} -D F=7 -D ON=1 -D IN=7 -D LN=7 -D UN=4 -D Fsq=49 -DUSE_HALF\"
            echo ./gcl_binary --input=$file --output=${file%.*}_72.bin  --options=\"${copt} -D F=7 -D ON=2 -D IN=8 -D LN=8 -D UN=5 -D Fsq=49 -DUSE_HALF\"
            echo ./gcl_binary --input=$file --output=${file%.*}_73.bin  --options=\"${copt} -D F=7 -D ON=3 -D IN=3 -D LN=2 -D UN=2 -D Fsq=49 -DUSE_HALF -D BASICE_REG\"
            echo ./gcl_binary --input=$file --output=${file%.*}_74.bin  --options=\"${copt} -D F=7 -D ON=4 -D IN=4 -D LN=3 -D UN=3 -D Fsq=49 -DUSE_HALF -D BASICE_REG\"
            echo ./gcl_binary --input=$file --output=${file%.*}_75.bin  --options=\"${copt} -D F=7 -D ON=5 -D IN=5 -D LN=4 -D UN=4 -D Fsq=49 -DUSE_HALF -D BASICE_REG\"
            echo ./gcl_binary --input=$file --output=${file%.*}_76.bin  --options=\"${copt} -D F=7 -D ON=6 -D IN=6 -D LN=5 -D UN=5 -D Fsq=49 -DUSE_HALF -D BASICE_REG\"
            echo ./gcl_binary --input=$file --output=${file%.*}_77.bin  --options=\"${copt} -D F=7 -D ON=7 -D IN=7 -D LN=6 -D UN=6 -D Fsq=49 -DUSE_HALF -D BASICE_REG\"
            echo ./gcl_binary --input=$file --output=${file%.*}_78.bin  --options=\"${copt} -D F=7 -D ON=8 -D IN=8 -D LN=7 -D UN=7 -D Fsq=49 -DUSE_HALF -D BASICE_REG\"

            echo ./gcl_binary --input=$file --output=${file%.*}_relu_71.bin  --options=\"${copt} -D F=7 -D ON=1 -D IN=7 -D LN=7 -D UN=6 -D Fsq=49 -DUSE_HALF -DUSE_RELU\"
            echo ./gcl_binary --input=$file --output=${file%.*}_relu_72.bin  --options=\"${copt} -D F=7 -D ON=2 -D IN=8 -D LN=8 -D UN=7 -D Fsq=49 -DUSE_HALF -DUSE_RELU\"
            echo ./gcl_binary --input=$file --output=${file%.*}_relu_73.bin  --options=\"${copt} -D F=7 -D ON=3 -D IN=3 -D LN=2 -D UN=2 -D Fsq=49 -DUSE_HALF -DUSE_RELU -D BASICE_REG\"
            echo ./gcl_binary --input=$file --output=${file%.*}_relu_74.bin  --options=\"${copt} -D F=7 -D ON=4 -D IN=4 -D LN=3 -D UN=3 -D Fsq=49 -DUSE_HALF -DUSE_RELU -D BASICE_REG\"
            echo ./gcl_binary --input=$file --output=${file%.*}_relu_75.bin  --options=\"${copt} -D F=7 -D ON=5 -D IN=5 -D LN=4 -D UN=4 -D Fsq=49 -DUSE_HALF -DUSE_RELU -D BASICE_REG\"
            echo ./gcl_binary --input=$file --output=${file%.*}_relu_76.bin  --options=\"${copt} -D F=7 -D ON=6 -D IN=6 -D LN=5 -D UN=5 -D Fsq=49 -DUSE_HALF -DUSE_RELU -D BASICE_REG\"
            echo ./gcl_binary --input=$file --output=${file%.*}_relu_77.bin  --options=\"${copt} -D F=7 -D ON=7 -D IN=7 -D LN=6 -D UN=6 -D Fsq=49 -DUSE_HALF -DUSE_RELU -D BASICE_REG\"
            echo ./gcl_binary --input=$file --output=${file%.*}_relu_78.bin  --options=\"${copt} -D F=7 -D ON=8 -D IN=8 -D LN=7 -D UN=7 -D Fsq=49 -DUSE_HALF -DUSE_RELU -D BASICE_REG\"

            echo ./gcl_binary --input=$file --output=${file%.*}_relu6_71.bin  --options=\"${copt} -D F=7 -D ON=1 -D IN=7 -D LN=7 -D UN=6 -D Fsq=49 -DUSE_HALF -DUSE_RELU6\"
            echo ./gcl_binary --input=$file --output=${file%.*}_relu6_72.bin  --options=\"${copt} -D F=7 -D ON=2 -D IN=8 -D LN=8 -D UN=7 -D Fsq=49 -DUSE_HALF -DUSE_RELU6\"
            echo ./gcl_binary --input=$file --output=${file%.*}_relu6_73.bin  --options=\"${copt} -D F=7 -D ON=3 -D IN=3 -D LN=2 -D UN=2 -D Fsq=49 -DUSE_HALF -DUSE_RELU6 -D BASICE_REG\"
            echo ./gcl_binary --input=$file --output=${file%.*}_relu6_74.bin  --options=\"${copt} -D F=7 -D ON=4 -D IN=4 -D LN=3 -D UN=3 -D Fsq=49 -DUSE_HALF -DUSE_RELU6 -D BASICE_REG\"
            echo ./gcl_binary --input=$file --output=${file%.*}_relu6_75.bin  --options=\"${copt} -D F=7 -D ON=5 -D IN=5 -D LN=4 -D UN=4 -D Fsq=49 -DUSE_HALF -DUSE_RELU6 -D BASICE_REG\"
            echo ./gcl_binary --input=$file --output=${file%.*}_relu6_76.bin  --options=\"${copt} -D F=7 -D ON=6 -D IN=6 -D LN=5 -D UN=5 -D Fsq=49 -DUSE_HALF -DUSE_RELU6 -D BASICE_REG\"
            echo ./gcl_binary --input=$file --output=${file%.*}_relu6_77.bin  --options=\"${copt} -D F=7 -D ON=7 -D IN=7 -D LN=6 -D UN=6 -D Fsq=49 -DUSE_HALF -DUSE_RELU6 -D BASICE_REG\"
            echo ./gcl_binary --input=$file --output=${file%.*}_relu6_78.bin  --options=\"${copt} -D F=7 -D ON=8 -D IN=8 -D LN=7 -D UN=7 -D Fsq=49 -DUSE_HALF -DUSE_RELU6 -D BASICE_REG\"

            echo ./gcl_binary --input=$file --output=${file%.*}_ncwh_31.bin  --options=\"${copt} -D F=3 -D ON=1 -D IN=3 -D LN=3 -D UN=2 -D Fsq=9 -DUSE_NCWH -DUSE_HALF\"
            echo ./gcl_binary --input=$file --output=${file%.*}_ncwh_32.bin  --options=\"${copt} -D F=3 -D ON=2 -D IN=4 -D LN=4 -D UN=3 -D Fsq=9 -DUSE_NCWH -DUSE_HALF\"
            echo ./gcl_binary --input=$file --output=${file%.*}_ncwh_33.bin  --options=\"${copt} -D F=3 -D ON=3 -D IN=5 -D LN=5 -D UN=4 -D Fsq=9 -DUSE_NCWH -DUSE_HALF\"
            echo ./gcl_binary --input=$file --output=${file%.*}_ncwh_34.bin  --options=\"${copt} -D F=3 -D ON=4 -D IN=6 -D LN=6 -D UN=5 -D Fsq=9 -DUSE_NCWH -DUSE_HALF\"
            echo ./gcl_binary --input=$file --output=${file%.*}_ncwh_35.bin  --options=\"${copt} -D F=3 -D ON=5 -D IN=7 -D LN=7 -D UN=6 -D Fsq=9 -DUSE_NCWH -DUSE_HALF\"
            echo ./gcl_binary --input=$file --output=${file%.*}_ncwh_36.bin  --options=\"${copt} -D F=3 -D ON=6 -D IN=6 -D LN=5 -D UN=5 -D Fsq=9 -DUSE_NCWH -DUSE_HALF -D BASICE_REG\"
            echo ./gcl_binary --input=$file --output=${file%.*}_ncwh_37.bin  --options=\"${copt} -D F=3 -D ON=7 -D IN=7 -D LN=6 -D UN=6 -D Fsq=9 -DUSE_NCWH -DUSE_HALF -D BASICE_REG\"
            echo ./gcl_binary --input=$file --output=${file%.*}_ncwh_38.bin  --options=\"${copt} -D F=3 -D ON=8 -D IN=8 -D LN=7 -D UN=7 -D Fsq=9 -DUSE_NCWH -DUSE_HALF -D BASICE_REG\"

            echo ./gcl_binary --input=$file --output=${file%.*}_relu_ncwh_31.bin  --options=\"${copt} -D F=3 -D ON=1 -D IN=3 -D LN=3 -D UN=2 -D Fsq=9  -DUSE_NCWH -DUSE_HALF -DUSE_RELU\"
            echo ./gcl_binary --input=$file --output=${file%.*}_relu_ncwh_32.bin  --options=\"${copt} -D F=3 -D ON=2 -D IN=4 -D LN=4 -D UN=3 -D Fsq=9  -DUSE_NCWH -DUSE_HALF -DUSE_RELU\"
            echo ./gcl_binary --input=$file --output=${file%.*}_relu_ncwh_33.bin  --options=\"${copt} -D F=3 -D ON=3 -D IN=5 -D LN=5 -D UN=4 -D Fsq=9  -DUSE_NCWH -DUSE_HALF -DUSE_RELU\"
            echo ./gcl_binary --input=$file --output=${file%.*}_relu_ncwh_34.bin  --options=\"${copt} -D F=3 -D ON=4 -D IN=6 -D LN=6 -D UN=5 -D Fsq=9  -DUSE_NCWH -DUSE_HALF -DUSE_RELU\"
            echo ./gcl_binary --input=$file --output=${file%.*}_relu_ncwh_35.bin  --options=\"${copt} -D F=3 -D ON=5 -D IN=7 -D LN=7 -D UN=6 -D Fsq=9  -DUSE_NCWH -DUSE_HALF -DUSE_RELU\"
            echo ./gcl_binary --input=$file --output=${file%.*}_relu_ncwh_36.bin  --options=\"${copt} -D F=3 -D ON=6 -D IN=6 -D LN=5 -D UN=5 -D Fsq=9  -DUSE_NCWH -DUSE_HALF -DUSE_RELU -D BASICE_REG\"
            echo ./gcl_binary --input=$file --output=${file%.*}_relu_ncwh_37.bin  --options=\"${copt} -D F=3 -D ON=7 -D IN=7 -D LN=6 -D UN=6 -D Fsq=9  -DUSE_NCWH -DUSE_HALF -DUSE_RELU -D BASICE_REG\"
            echo ./gcl_binary --input=$file --output=${file%.*}_relu_ncwh_38.bin  --options=\"${copt} -D F=3 -D ON=8 -D IN=8 -D LN=7 -D UN=7 -D Fsq=9  -DUSE_NCWH -DUSE_HALF -DUSE_RELU -D BASICE_REG\"


            echo ./gcl_binary --input=$file --output=${file%.*}_relu6_ncwh_31.bin  --options=\"${copt} -D F=3 -D ON=1 -D IN=3 -D LN=3 -D UN=2 -D Fsq=9  -DUSE_NCWH -DUSE_HALF -DUSE_RELU6\"
            echo ./gcl_binary --input=$file --output=${file%.*}_relu6_ncwh_32.bin  --options=\"${copt} -D F=3 -D ON=2 -D IN=4 -D LN=4 -D UN=3 -D Fsq=9  -DUSE_NCWH -DUSE_HALF -DUSE_RELU6\"
            echo ./gcl_binary --input=$file --output=${file%.*}_relu6_ncwh_33.bin  --options=\"${copt} -D F=3 -D ON=3 -D IN=5 -D LN=5 -D UN=4 -D Fsq=9  -DUSE_NCWH -DUSE_HALF -DUSE_RELU6\"
            echo ./gcl_binary --input=$file --output=${file%.*}_relu6_ncwh_34.bin  --options=\"${copt} -D F=3 -D ON=4 -D IN=6 -D LN=6 -D UN=5 -D Fsq=9  -DUSE_NCWH -DUSE_HALF -DUSE_RELU6\"
            echo ./gcl_binary --input=$file --output=${file%.*}_relu6_ncwh_35.bin  --options=\"${copt} -D F=3 -D ON=5 -D IN=7 -D LN=7 -D UN=6 -D Fsq=9  -DUSE_NCWH -DUSE_HALF -DUSE_RELU6\"
            echo ./gcl_binary --input=$file --output=${file%.*}_relu6_ncwh_36.bin  --options=\"${copt} -D F=3 -D ON=6 -D IN=6 -D LN=5 -D UN=5 -D Fsq=9  -DUSE_NCWH -DUSE_HALF -DUSE_RELU6 -D BASICE_REG\"
            echo ./gcl_binary --input=$file --output=${file%.*}_relu6_ncwh_37.bin  --options=\"${copt} -D F=3 -D ON=7 -D IN=7 -D LN=6 -D UN=6 -D Fsq=9  -DUSE_NCWH -DUSE_HALF -DUSE_RELU6 -D BASICE_REG\"
            echo ./gcl_binary --input=$file --output=${file%.*}_relu6_ncwh_38.bin  --options=\"${copt} -D F=3 -D ON=8 -D IN=8 -D LN=7 -D UN=7 -D Fsq=9  -DUSE_NCWH -DUSE_HALF -DUSE_RELU6 -D BASICE_REG\"


            echo ./gcl_binary --input=$file --output=${file%.*}_ncwh_51.bin  --options=\"${copt} -D F=5 -D ON=1 -D IN=5 -D LN=5 -D UN=4 -D Fsq=25 -DUSE_NCWH -DUSE_HALF\"
            echo ./gcl_binary --input=$file --output=${file%.*}_ncwh_52.bin  --options=\"${copt} -D F=5 -D ON=2 -D IN=6 -D LN=6 -D UN=5 -D Fsq=25 -DUSE_NCWH -DUSE_HALF\"
            echo ./gcl_binary --input=$file --output=${file%.*}_ncwh_53.bin  --options=\"${copt} -D F=5 -D ON=3 -D IN=7 -D LN=7 -D UN=6 -D Fsq=25 -DUSE_NCWH -DUSE_HALF\"
            echo ./gcl_binary --input=$file --output=${file%.*}_ncwh_54.bin  --options=\"${copt} -D F=5 -D ON=4 -D IN=4 -D LN=3 -D UN=3 -D Fsq=25 -DUSE_NCWH -DUSE_HALF -D BASICE_REG\"
            echo ./gcl_binary --input=$file --output=${file%.*}_ncwh_55.bin  --options=\"${copt} -D F=5 -D ON=5 -D IN=5 -D LN=4 -D UN=4 -D Fsq=25 -DUSE_NCWH -DUSE_HALF -D BASICE_REG\"
            echo ./gcl_binary --input=$file --output=${file%.*}_ncwh_56.bin  --options=\"${copt} -D F=5 -D ON=6 -D IN=6 -D LN=5 -D UN=5 -D Fsq=25 -DUSE_NCWH -DUSE_HALF -D BASICE_REG\"
            echo ./gcl_binary --input=$file --output=${file%.*}_ncwh_57.bin  --options=\"${copt} -D F=5 -D ON=7 -D IN=7 -D LN=6 -D UN=6 -D Fsq=25 -DUSE_NCWH -DUSE_HALF -D BASICE_REG\"
            echo ./gcl_binary --input=$file --output=${file%.*}_ncwh_58.bin  --options=\"${copt} -D F=5 -D ON=8 -D IN=8 -D LN=7 -D UN=7 -D Fsq=25 -DUSE_NCWH -DUSE_HALF -D BASICE_REG\"

            echo ./gcl_binary --input=$file --output=${file%.*}_relu_ncwh_51.bin  --options=\"${copt} -D F=5 -D ON=1 -D IN=5 -D LN=5 -D UN=4 -D Fsq=25 -DUSE_NCWH -DUSE_HALF -DUSE_RELU\"
            echo ./gcl_binary --input=$file --output=${file%.*}_relu_ncwh_52.bin  --options=\"${copt} -D F=5 -D ON=2 -D IN=6 -D LN=6 -D UN=5 -D Fsq=25 -DUSE_NCWH -DUSE_HALF -DUSE_RELU\"
            echo ./gcl_binary --input=$file --output=${file%.*}_relu_ncwh_53.bin  --options=\"${copt} -D F=5 -D ON=3 -D IN=7 -D LN=7 -D UN=6 -D Fsq=25 -DUSE_NCWH -DUSE_HALF -DUSE_RELU\"
            echo ./gcl_binary --input=$file --output=${file%.*}_relu_ncwh_54.bin  --options=\"${copt} -D F=5 -D ON=4 -D IN=4 -D LN=3 -D UN=3 -D Fsq=25 -DUSE_NCWH -DUSE_HALF -DUSE_RELU -D BASICE_REG\"
            echo ./gcl_binary --input=$file --output=${file%.*}_relu_ncwh_55.bin  --options=\"${copt} -D F=5 -D ON=5 -D IN=5 -D LN=4 -D UN=4 -D Fsq=25 -DUSE_NCWH -DUSE_HALF -DUSE_RELU -D BASICE_REG\"
            echo ./gcl_binary --input=$file --output=${file%.*}_relu_ncwh_56.bin  --options=\"${copt} -D F=5 -D ON=6 -D IN=6 -D LN=5 -D UN=5 -D Fsq=25 -DUSE_NCWH -DUSE_HALF -DUSE_RELU -D BASICE_REG\"
            echo ./gcl_binary --input=$file --output=${file%.*}_relu_ncwh_57.bin  --options=\"${copt} -D F=5 -D ON=7 -D IN=7 -D LN=6 -D UN=6 -D Fsq=25 -DUSE_NCWH -DUSE_HALF -DUSE_RELU -D BASICE_REG\"
            echo ./gcl_binary --input=$file --output=${file%.*}_relu_ncwh_58.bin  --options=\"${copt} -D F=5 -D ON=8 -D IN=8 -D LN=7 -D UN=7 -D Fsq=25 -DUSE_NCWH -DUSE_HALF -DUSE_RELU -D BASICE_REG\"


            echo ./gcl_binary --input=$file --output=${file%.*}_relu6_ncwh_51.bin  --options=\"${copt} -D F=5 -D ON=1 -D IN=5 -D LN=5 -D UN=4 -D Fsq=25 -DUSE_NCWH -DUSE_HALF -DUSE_RELU6\"
            echo ./gcl_binary --input=$file --output=${file%.*}_relu6_ncwh_52.bin  --options=\"${copt} -D F=5 -D ON=2 -D IN=6 -D LN=6 -D UN=5 -D Fsq=25 -DUSE_NCWH -DUSE_HALF -DUSE_RELU6\"
            echo ./gcl_binary --input=$file --output=${file%.*}_relu6_ncwh_53.bin  --options=\"${copt} -D F=5 -D ON=3 -D IN=7 -D LN=7 -D UN=6 -D Fsq=25 -DUSE_NCWH -DUSE_HALF -DUSE_RELU6\"
            echo ./gcl_binary --input=$file --output=${file%.*}_relu6_ncwh_54.bin  --options=\"${copt} -D F=5 -D ON=4 -D IN=4 -D LN=3 -D UN=3 -D Fsq=25 -DUSE_NCWH -DUSE_HALF -DUSE_RELU6 -D BASICE_REG\"
            echo ./gcl_binary --input=$file --output=${file%.*}_relu6_ncwh_55.bin  --options=\"${copt} -D F=5 -D ON=5 -D IN=5 -D LN=4 -D UN=4 -D Fsq=25 -DUSE_NCWH -DUSE_HALF -DUSE_RELU6 -D BASICE_REG\"
            echo ./gcl_binary --input=$file --output=${file%.*}_relu6_ncwh_56.bin  --options=\"${copt} -D F=5 -D ON=6 -D IN=6 -D LN=5 -D UN=5 -D Fsq=25 -DUSE_NCWH -DUSE_HALF -DUSE_RELU6 -D BASICE_REG\"
            echo ./gcl_binary --input=$file --output=${file%.*}_relu6_ncwh_57.bin  --options=\"${copt} -D F=5 -D ON=7 -D IN=7 -D LN=6 -D UN=6 -D Fsq=25 -DUSE_NCWH -DUSE_HALF -DUSE_RELU6 -D BASICE_REG\"
            echo ./gcl_binary --input=$file --output=${file%.*}_relu6_ncwh_58.bin  --options=\"${copt} -D F=5 -D ON=8 -D IN=8 -D LN=7 -D UN=7 -D Fsq=25 -DUSE_NCWH -DUSE_HALF -DUSE_RELU6 -D BASICE_REG\"
            fi
        fi
    done



