#!/bin/bash

targets=("android-aarch64" "android-armv7" "android-x86_64" \
    "ios-aarch64" "ios-armv7" \
    "linux-x86_64" "linux-x86_64_avx2" "linux-aarch64" "linux-arm_himix100" "linux-armv7_blank" "linux-arm_musleabi" \
    "windows-x86_64" "windows-x86_64_avx2" \
    "macos-x86_64" "macos-x86_64_avx2")

print_targets() {
    for((i=0; i<${#targets[@]}; i++)) do
        element=${targets[i]};
        cat <<EOF
           ${i} = ${element}
EOF
    done

}

map_target() {
    target=$1
    for((i=0; i<${#targets[@]}; i++)) do
        if [[ "${target}" == "$i" ]]; then
            target=${targets[i]}
            break;
        fi
    done
    echo ${target}
}

check_target() {
    if [[ ${1} == "" ]]; then
        return 0;
    fi
    find_target=false
    for element in ${targets[@]}; do
        if [[ ${element} == ${1} ]]; then
            find_target=true
        fi
    done
    if [[ ${find_target} == false ]]; then
        echo "[ERROR] not support to build on target ${1}, currently only support theses targets:"
        print_targets
        exit 1;
    fi
}

check_getopt() {
    getopt --test
    if [[ "$?" != "4" ]]; then
        echo -e "[ERROR] you are using BSD getopt, not GNU getopt. If you are runing on Mac, please use this command to install gnu-opt.\n    brew install gnu-getopt && brew link --force gnu-getopt"
        exit 1
    fi
}
