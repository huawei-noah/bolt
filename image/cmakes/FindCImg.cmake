find_path(CImg_INCLUDE_DIR NAMES CImg.h 
          HINTS ${CImg_ROOT}/include $ENV{CImg_ROOT}/include ${CImg_ROOT} $ENV{CImg_ROOT})

if (CImg_INCLUDE_DIR)
    set(CImg_FOUND true)
endif (CImg_INCLUDE_DIR)

if (CImg_FOUND)
    message(STATUS "Found CImg.h: ${CImg_INCLUDE_DIR}")
else (CImg_FOUND)
    message(FATAL_ERROR "Could not find CImg header")
endif (CImg_FOUND)
