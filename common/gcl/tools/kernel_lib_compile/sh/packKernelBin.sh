#pack all kernelbin#
#build headfile and lib.a#
BINHEAD=gcl_kernel_binmap.h
Head=libkernelbin.h
UpperHead=$(echo libkernelbin | tr '[a-z]' '[A-Z]')
echo "#ifndef _${UpperHead}_H" > $incPath/$Head
echo "#define _${UpperHead}_H" >> $incPath/$Head
echo >> $incPath/$Head
echo "#include \""gcl_kernel_binmap.h"\"" >> $incPath/$Head
echo >> $incPath/$Head

deviceNamesAll=$(awk '{print $1}' ${dNameFile})
deviceNum=${#deviceNamesAll[@]}
for((i=1;i<deviceNum;i++)); do
    dname=${deviceNamesAll[$i]};
    dnameS=0
    for((j=1;j<i;j++)); do
        dnameJ=${deviceNamesAll[$j]};
        if [ "$dnameJ" = "$dname" ];then
            dnameS=1
        fi
    done
    
    if [ $dnameS -eq 0 ]; then
        InlineHead=inline_${dname}.h
        InlineCpp=inline_${dname}.cpp
        UpperInlineHead=$(echo inline_${dname} | tr '[a-z]' '[A-Z]')
        echo "class ${dname} : public gcl_kernel_binmap {" >> $incPath/$Head
        echo "public:" >> $incPath/$Head
        echo "    ${dname}() {" >> $incPath/$Head
        echo "        loadKernelBin();" >> $incPath/$Head
        echo "    }" >> $incPath/$Head
        echo "    void loadKernelBin();" >> $incPath/$Head
        echo "};" >> $incPath/$Head
        echo >> $incPath/$Head
        echo >> $incPath/$Head
        echo "REGISTER_GCLKERNELMAP(${dname});" >> $incPath/$Head
        echo >> $incPath/$Head

        echo "#ifndef _${UpperInlineHead}_H" > $srcPath/$InlineHead
        echo "#define _${UpperInlineHead}_H" >> $srcPath/$InlineHead
        echo "#include \""types.h"\"" >> $srcPath/$InlineHead
        echo >> $srcPath/$InlineHead

        echo "#include \""${Head}"\"" > $srcPath/$InlineCpp
        echo "#include \""${InlineHead}"\"" >> $srcPath/$InlineCpp
        echo >> $srcPath/$InlineCpp
        echo "void ${dname}::loadKernelBin() {" >> $srcPath/$InlineCpp

        cpp=.cpp
        for file in `ls $binPath/$dname`
            do
            var=${file}
            var=${var//.*/}
            var=${dname}_${var}
            ./bin2char bin/$dname/$file src/${var}${cpp} ${dname}
            echo "extern CU32 "${var}"_len;" >> $srcPath/$InlineHead
            echo "extern CU8  "${var}"[];" >> $srcPath/$InlineHead
            echo >> $srcPath/$InlineHead
            echo "    put(\""${var}"\", {"${var}", "${var}"_len});" >> $srcPath/$InlineCpp
        done

        echo >> $srcPath/$InlineHead
        echo "#endif" >> $srcPath/$InlineHead
        echo "}" >> $srcPath/$InlineCpp
    fi
done
echo "#endif" >> $incPath/$Head

#get headfile
kernelBin=${BOLT_ROOT}/gcl/kernelBin/
rm -rf ${kernelBin}
mkdir ${kernelBin}
mkdir ${kernelBin}/include/
cp ${incPath}/*.h ${kernelBin}/include/
