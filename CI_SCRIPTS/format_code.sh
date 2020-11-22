#!/bin/bash

script_name=$0
script_abs=$(readlink -f "$0")
script_dir=$(dirname $script_abs)

fileSuffix=(h hpp c cpp cl)

cd ${script_dir}/../
format() {
    file=$1
    echo "format: $file" 
    #/data/opt/uncrustify-master/build/uncrustify -c /data/opt/uncrustify-master/forUncrustifySources.cfg -f $file > tmp.cpp
    #sed -i "s/\/\/ /\/\//g" ./tmp.cpp
    #sed -i "s/\/\//\/\/ /g" ./tmp.cpp
    #clang-format -i tmp.cpp
    #cp tmp.cpp $file
    #rm tmp.cpp
    clang-format -i $file
}

format_all() {
    dirs=(inference common model_tools compute kit)
    for suffix in ${fileSuffix[*]}
    do
        for dir in  ${dirs[*]}
        do
            for file in `find $dir -name "*.$suffix"`
            do
                format $file
            done
        done
    done
}

format_change() {
    key=$1
    files=`git status | grep "${key}" | sed s/[[:space:]]//g | sed s/：/:/g | cut -d ":" -f 2`
    for file in ${files[*]}
    do
        fresh=false
        for suffix in ${fileSuffix[*]}
        do
            if [[ $file == *.${suffix} ]]; then
                fresh=true
            fi
        done
        if [[ $fresh == true ]]; then
            format $file
        fi
    done
}

format_change "modified:"
format_change "修改："
