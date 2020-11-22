#!/bin/bash

fun_gen_in_two_arrs() {
    rm -rf ./single_combinations.txt
    #touch ./single_combinations.txt
    local _firstArr=(`echo $1|cut -d " " --output-delimiter=" " -f 1-`)
    local _firstArrLen=${#_firstArr[@]}
    local _secondArr=(`echo $2|cut -d " " --output-delimiter=" " -f 1-`)
    local _secondArrLen=${#_secondArr[@]}
    index=0
    for ((i=0;i<_firstArrLen;i++))
    do
        for ((j=0;j<_secondArrLen;j++))
        do
            elem1=${_firstArr[$i]}
            elem2=${_secondArr[$j]}
            combine_str=$elem1"--"$elem2
            echo $combine_str >> ./single_combinations.txt
            let index+=1
        done
    done
}

rm -rf ./final_combinations.txt
while read line
do
    if [[ ${line} =~ ^#.* ]]; then
        continue
    fi
    original_strs=()
    original_index=0
    for i in $(echo $line| tr "&" "\n")
    do
        original_strs[$original_index]=$i
        let original_index+=1
    done

    for i in "${!original_strs[@]}";
    do
        sub_str=${original_strs[$i]}
        if [ $i == 0 ]
        then
            rm -rf ./single_combinations.txt
            for j in $(echo $sub_str| tr ";" "\n")
            do
                echo $j >> ./single_combinations.txt
            done
        else
            sub_firstArr=()
            sub_firstIndex=0
            for line in `cat ./single_combinations.txt`
            do
                sub_firstArr[$sub_firstIndex]=$line
                let sub_firstIndex+=1
            done
            sub_secondArr=($(echo "$sub_str"| tr ";" "\n"))
            fun_gen_in_two_arrs "$(echo ${sub_firstArr[@]})" "$(echo ${sub_secondArr[@]})"
        fi
    done

    cat ./single_combinations.txt >> ./final_combinations.txt
done < $1
rm -rf ./single_combinations.txt
