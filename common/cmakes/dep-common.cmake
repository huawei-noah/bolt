
function(dependency_folder_not_empty PATH_ RESULT_)
    set(res_ FALSE)
    if(EXISTS ${PATH_})
        file(GLOB files_ "${PATH_}/*")
        if(files_)
            set(res_ TRUE)
        endif()
        unset(files_)
    endif()
    set(${RESULT_} ${res_} PARENT_SCOPE)
endfunction()

function(file_exists STATUS_ FILE_ HASH_MD5_)
    set(${STATUS_} FALSE PARENT_SCOPE)
    if(EXISTS ${FILE_})
        file(MD5 ${FILE_} hash_)
        if(${hash_} STREQUAL ${HASH_MD5_})
            set(${STATUS_} TRUE PARENT_SCOPE)
        else()
            message(FATAL_ERROR "${FILE_} has hash ${hash_}")
        endif()
    endif()
endfunction()


function(dependency_download_and_unzip URL_ HASH_MD5_ FILE_ UNZIPPED_DIR_ ONLY_CHECK_ RET_CODE_)
    get_filename_component(workdir_ ${FILE_} DIRECTORY)
    set(unzipped_directory ${workdir_}/unzipped)

    dependency_folder_not_empty(${unzipped_directory} unzip_exists_)
    if(NOT ${unzip_exists_})
        file_exists(archive_is_valid_ ${FILE_} ${HASH_MD5_})
        if(NOT ${archive_is_valid_})
            if(ONLY_CHECK_)
                set(${RET_CODE_} FALSE PARENT_SCOPE)
                return()
            endif()

            message(STATUS "Downloading: ${URL_}")
            file(DOWNLOAD ${URL_} ${FILE_}
                SHOW_PROGRESS
                EXPECTED_MD5 ${HASH_MD5_}
                STATUS ret_
            )
            if(NOT ret_)
                message(SEND_ERROR "Failed to download: ${URL_}")
                set(${RET_CODE_} FALSE PARENT_SCOPE)
                return()
            endif()
        endif()

        message(STATUS "Unzipping: ${FILE_}")
        file(MAKE_DIRECTORY ${unzipped_directory})
        execute_process(COMMAND "${CMAKE_COMMAND}" -E tar -xzf ${FILE_}
            WORKING_DIRECTORY ${unzipped_directory}
            RESULT_VARIABLE ret_
        )
        if(ret_)
            message(SEND_ERROR "Failed to extract: ${FILE_}")
            set(${RET_CODE_} FALSE PARENT_SCOPE)
            return()
        endif()
    endif()

    set(${UNZIPPED_DIR_} ${unzipped_directory} PARENT_SCOPE)
    set(${RET_CODE_} TRUE PARENT_SCOPE)
endfunction()

function(dependency_download)
    set(oneValueArgs
        NAME            # Short dependency name used for cache variables
        VERBOSE_NAME    # Verbose name for messages and tooltips
        URL             # Url for downloading.
        HASH_MD5        # Hash of downloaded file
        FILE_NAME       # specify downloaded file name
        PREFIX          # [Optional] Additonal path prefix after extracting archive
    )
    cmake_parse_arguments(DEPENDENCY "" "${oneValueArgs}" "" ${ARGN})

    set(file_name_ ${DEPENDENCY_FILE_NAME})
    
    string(TOLOWER ${DEPENDENCY_NAME} folder_name_)
    set(dependency_file_ "${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/sources/${folder_name_}/${file_name_}")

    dependency_download_and_unzip(${DEPENDENCY_URL} ${DEPENDENCY_HASH_MD5} ${dependency_file_} unzipped_location_ FALSE dependency_is_downloaded_)
    if(${dependency_is_downloaded_})
        set(root_ ${unzipped_location_})
        if(DEPENDENCY_PREFIX)
            set(root_ ${root_}/${DEPENDENCY_PREFIX})
        endif()
    endif()

    if(NOT EXISTS ${root_})
        message(FATAL_ERROR "${DEPENDENCY_VERBOSE_NAME} is not found!")
    endif()

    set(${DEPENDENCY_NAME}_FOUND_ROOT ${root_} CACHE PATH "Found ${DEPENDENCY_VERBOSE_NAME} root directory. Read-only variable." FORCE)
endfunction()
