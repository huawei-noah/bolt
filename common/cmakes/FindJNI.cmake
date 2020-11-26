find_path(JNI_INCLUDE_DIR NAMES jni.h HINTS $ENV{JNI_ROOT}/include ${JNI_ROOT}/include $ENV{JNI_ROOT}/include/linux ${JNI_ROOT}/include/linux)
find_path(JNI_MD_INCLUDE_DIR NAMES jni_md.h HINTS $ENV{JNI_ROOT}/include ${JNI_ROOT}/include $ENV{JNI_ROOT}/include/linux ${JNI_ROOT}/include/linux)

if (JNI_INCLUDE_DIR AND JNI_MD_INCLUDE_DIR)
    set(JNI_FOUND true)
else (JNI_INCLUDE_DIR AND JNI_MD_INCLUDE_DIR)
    set(JNI_FOUND false)
endif (JNI_INCLUDE_DIR AND JNI_MD_INCLUDE_DIR)

if (JNI_FOUND)
    message(STATUS "Found jni.h: ${JNI_INCLUDE_DIR}")
else (JNI_FOUND)
    message(WARNING "WARNING: can not find jni.h/jni_md.h in <JNI_ROOT>/include or <JNI_ROOT>/include/linux directory, so can not use Java API. If you want to use Java API, please set shell or cmake environment variable JNI_ROOT.")
endif (JNI_FOUND)
