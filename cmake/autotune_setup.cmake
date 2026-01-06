# Prepare local autotune environment (directories + symlinks + optional stubs).

if(NOT DEFINED AUTOTUNE_DIR)
  message(FATAL_ERROR "AUTOTUNE_DIR is required")
endif()
if(NOT DEFINED HALIDE_PREFIX)
  message(FATAL_ERROR "HALIDE_PREFIX is required")
endif()

set(AUTOSCHED_BIN "${AUTOTUNE_DIR}/autosched_bin")
set(HALIDE_DIST "${AUTOTUNE_DIR}/halide_dist")
set(HALIDE_TOOLS_DIR "${AUTOTUNE_DIR}/tools")
set(HALIDE_BUILD_DIR "${AUTOTUNE_DIR}/halide_build")
set(AUTOTUNE_BIN "${AUTOTUNE_DIR}/bin")
set(SAMPLES_DIR "${AUTOTUNE_DIR}/samples")

file(MAKE_DIRECTORY "${AUTOTUNE_DIR}")
file(MAKE_DIRECTORY "${AUTOSCHED_BIN}")
file(MAKE_DIRECTORY "${HALIDE_DIST}")
file(MAKE_DIRECTORY "${HALIDE_TOOLS_DIR}")
file(MAKE_DIRECTORY "${HALIDE_BUILD_DIR}")
file(MAKE_DIRECTORY "${AUTOTUNE_BIN}")
file(MAKE_DIRECTORY "${SAMPLES_DIR}")

function(prism_link src dst)
  if(EXISTS "${dst}" OR IS_SYMLINK "${dst}")
    file(REMOVE "${dst}")
  endif()
  execute_process(
    COMMAND "${CMAKE_COMMAND}" -E create_symlink "${src}" "${dst}"
    RESULT_VARIABLE _rv
  )
  if(NOT _rv EQUAL 0)
    message(WARNING "Failed to create symlink: ${dst} -> ${src}")
  endif()
endfunction()

# Halide distribution layout expected by the autotune scripts.
prism_link("${HALIDE_PREFIX}/include" "${HALIDE_DIST}/include")
prism_link("${HALIDE_PREFIX}/share/tools" "${HALIDE_DIST}/tools")

# Loop scripts from the installed Halide tools directory.
prism_link("${HALIDE_PREFIX}/share/tools/adams2019_autotune_loop.sh"
           "${HALIDE_TOOLS_DIR}/adams2019_autotune_loop.sh")
set(_anderson_loop_src "${HALIDE_PREFIX}/share/tools/anderson2021_autotune_loop.sh")
set(_anderson_loop_dst "${HALIDE_TOOLS_DIR}/anderson2021_autotune_loop.sh")
if(EXISTS "${_anderson_loop_src}")
  if(EXISTS "${_anderson_loop_dst}" OR IS_SYMLINK "${_anderson_loop_dst}")
    file(REMOVE "${_anderson_loop_dst}")
  endif()
  file(COPY_FILE "${_anderson_loop_src}" "${_anderson_loop_dst}")
  file(READ "${_anderson_loop_dst}" _anderson_loop_content)
  if(CMAKE_HOST_SYSTEM_NAME STREQUAL "Darwin")
    if(EXISTS "/usr/bin/time")
      find_program(_gtime_bin gtime)
      if(_gtime_bin)
        string(REPLACE "/bin/time" "gtime"
               _anderson_loop_content "${_anderson_loop_content}")
        string(REPLACE "/usr/bin/time" "gtime"
               _anderson_loop_content "${_anderson_loop_content}")
      else()
        string(REPLACE "/bin/time -f 'Compile time (s): %e'" "/usr/bin/time -p"
               _anderson_loop_content "${_anderson_loop_content}")
        string(REPLACE "/usr/bin/time -f 'Compile time (s): %e'" "/usr/bin/time -p"
               _anderson_loop_content "${_anderson_loop_content}")
      endif()
    endif()
  endif()
  string(REPLACE
    [==[    LIBPNG_CFLAGS=$(libpng-config --cflags)]==]
    [==[    LIBPNG_CFLAGS="${PRISM_PNG_CFLAGS:-$(libpng-config --cflags)}"]==]
    _anderson_loop_content "${_anderson_loop_content}")
  string(REPLACE
    [==[    LIBPNG_LIBS=$(libpng-config --ldflags)]==]
    [==[    LIBPNG_LIBS="${PRISM_PNG_LIBS:-$(libpng-config --ldflags)}"
    JPEG_CFLAGS="${PRISM_JPEG_CFLAGS:-}"
    JPEG_LIBS="${PRISM_JPEG_LIBS:--ljpeg}"
    FRAMEWORKS="${PRISM_AUTOTUNE_FRAMEWORKS:-}"]==]
    _anderson_loop_content "${_anderson_loop_content}")
  string(REPLACE
    [==[        ${LIBPNG_CFLAGS} \]==]
    [==[        ${LIBPNG_CFLAGS} \
        ${JPEG_CFLAGS} \]==]
    _anderson_loop_content "${_anderson_loop_content}")
  string(REPLACE
    [==[        -ljpeg ${LIBPNG_LIBS} -ldl -lpthread"]==]
    [==[        ${JPEG_LIBS} ${LIBPNG_LIBS} -ldl -lpthread ${FRAMEWORKS}"]==]
    _anderson_loop_content "${_anderson_loop_content}")
  string(REPLACE "-std=c++11" "-std=c++17"
         _anderson_loop_content "${_anderson_loop_content}")
  file(WRITE "${_anderson_loop_dst}" "${_anderson_loop_content}")
  file(CHMOD "${_anderson_loop_dst}"
    PERMISSIONS
      OWNER_READ OWNER_WRITE OWNER_EXECUTE
      GROUP_READ GROUP_EXECUTE
      WORLD_READ WORLD_EXECUTE
  )
else()
  message(WARNING "Anderson2021 loop script not found at ${_anderson_loop_src}")
endif()
unset(_anderson_loop_src)
unset(_anderson_loop_dst)
unset(_anderson_loop_content)
unset(_gtime_bin)

if(DEFINED ANDERSON2021_SCRIPTS_DIR AND EXISTS "${ANDERSON2021_SCRIPTS_DIR}/utils.sh")
  set(_anderson_scripts "${HALIDE_TOOLS_DIR}/scripts")
  file(MAKE_DIRECTORY "${_anderson_scripts}")
  file(GLOB _script_files "${ANDERSON2021_SCRIPTS_DIR}/*.sh")
  foreach(_script IN LISTS _script_files)
    file(COPY "${_script}" DESTINATION "${_anderson_scripts}")
  endforeach()
  file(GLOB _copied "${_anderson_scripts}/*.sh")
  foreach(_script IN LISTS _copied)
    file(CHMOD "${_script}"
      PERMISSIONS
        OWNER_READ OWNER_WRITE OWNER_EXECUTE
        GROUP_READ GROUP_EXECUTE
        WORLD_READ WORLD_EXECUTE
    )
  endforeach()
else()
  message(WARNING "Anderson2021 scripts not found; set ANDERSON2021_SCRIPTS_DIR to Halide source autoschedulers/anderson2021/scripts")
endif()

# Autoscheduler tools + plugins.
prism_link("${HALIDE_PREFIX}/bin/get_host_target" "${AUTOSCHED_BIN}/get_host_target")
prism_link("${HALIDE_PREFIX}/bin/featurization_to_sample" "${AUTOSCHED_BIN}/featurization_to_sample")
prism_link("${HALIDE_PREFIX}/bin/adams2019_retrain_cost_model" "${AUTOSCHED_BIN}/adams2019_retrain_cost_model")
prism_link("${HALIDE_PREFIX}/bin/anderson2021_retrain_cost_model" "${AUTOSCHED_BIN}/anderson2021_retrain_cost_model")
prism_link("${HALIDE_PREFIX}/lib/libautoschedule_adams2019.so" "${AUTOSCHED_BIN}/libautoschedule_adams2019.so")
prism_link("${HALIDE_PREFIX}/lib/libautoschedule_anderson2021.so" "${AUTOSCHED_BIN}/libautoschedule_anderson2021.so")

# Fake Halide build layout for Anderson2021 (expects HALIDE_BUILD_DIR layout).
set(_build_tools "${HALIDE_BUILD_DIR}/tools")
set(_build_autosched "${HALIDE_BUILD_DIR}/src/autoschedulers/anderson2021")
file(MAKE_DIRECTORY "${_build_tools}")
file(MAKE_DIRECTORY "${_build_autosched}")

prism_link("${HALIDE_PREFIX}/bin/get_host_target" "${_build_autosched}/get_host_target")
prism_link("${HALIDE_PREFIX}/bin/featurization_to_sample" "${_build_autosched}/featurization_to_sample")
prism_link("${HALIDE_PREFIX}/bin/anderson2021_retrain_cost_model" "${_build_autosched}/anderson2021_retrain_cost_model")
prism_link("${HALIDE_PREFIX}/lib/libautoschedule_anderson2021.so" "${_build_autosched}/libautoschedule_anderson2021.so")

set(_rungen_src "${HALIDE_PREFIX}/share/tools/RunGenMain.cpp")
set(_rungen_obj "${_build_tools}/RunGenMain.cpp.o")
if(EXISTS "${_rungen_src}")
  if(NOT DEFINED CMAKE_CXX_COMPILER OR CMAKE_CXX_COMPILER STREQUAL "")
    set(CMAKE_CXX_COMPILER "c++")
  endif()
  if(NOT EXISTS "${_rungen_obj}")
    execute_process(
      COMMAND "${CMAKE_CXX_COMPILER}"
              -std=c++17 -O3
              -I "${HALIDE_PREFIX}/include"
              -c "${_rungen_src}"
              -o "${_rungen_obj}"
      RESULT_VARIABLE _rv
    )
    if(NOT _rv EQUAL 0)
      message(WARNING "Failed to compile RunGenMain.cpp.o at ${_rungen_obj}")
    endif()
  endif()
else()
  message(WARNING "RunGenMain.cpp not found at ${_rungen_src}")
endif()

set(_nvidia_smi_export "")
if(DEFINED CREATE_FAKE_NVIDIA_SMI AND CREATE_FAKE_NVIDIA_SMI)
  set(_fake "${AUTOTUNE_BIN}/nvidia-smi")
  file(WRITE "${_fake}"
"#!/usr/bin/env bash
# Minimal stub for scripts that only query GPU count.
if [[ \"$1\" == \"--query-gpu=name\" ]]; then
  echo \"FakeGPU\"
  exit 0
fi
echo \"FakeGPU\"
")
  file(CHMOD "${_fake}"
    PERMISSIONS
      OWNER_READ OWNER_WRITE OWNER_EXECUTE
      GROUP_READ GROUP_EXECUTE
      WORLD_READ WORLD_EXECUTE
  )
  set(_nvidia_smi_export "export NVIDIA_SMI_BIN=\"${AUTOTUNE_BIN}/nvidia-smi\"\n")
endif()

set(_env_file "${AUTOTUNE_DIR}/env.sh")
if(NOT DEFINED PRISM_AUTOTUNE_PNG_CFLAGS)
  set(PRISM_AUTOTUNE_PNG_CFLAGS "")
endif()
if(NOT DEFINED PRISM_AUTOTUNE_PNG_LIBS)
  set(PRISM_AUTOTUNE_PNG_LIBS "")
endif()
if(NOT DEFINED PRISM_AUTOTUNE_JPEG_CFLAGS)
  set(PRISM_AUTOTUNE_JPEG_CFLAGS "")
endif()
if(NOT DEFINED PRISM_AUTOTUNE_JPEG_LIBS)
  set(PRISM_AUTOTUNE_JPEG_LIBS "")
endif()
if(NOT DEFINED PRISM_AUTOTUNE_FRAMEWORKS)
  set(PRISM_AUTOTUNE_FRAMEWORKS "")
endif()
set(_prism_png_cflags "${PRISM_AUTOTUNE_PNG_CFLAGS}")
set(_prism_png_libs "${PRISM_AUTOTUNE_PNG_LIBS}")
set(_prism_jpeg_cflags "${PRISM_AUTOTUNE_JPEG_CFLAGS}")
set(_prism_jpeg_libs "${PRISM_AUTOTUNE_JPEG_LIBS}")
set(_prism_frameworks "${PRISM_AUTOTUNE_FRAMEWORKS}")
string(REPLACE "\\ " " " _prism_png_cflags "${_prism_png_cflags}")
string(REPLACE "\\ " " " _prism_png_libs "${_prism_png_libs}")
string(REPLACE "\\ " " " _prism_jpeg_cflags "${_prism_jpeg_cflags}")
string(REPLACE "\\ " " " _prism_jpeg_libs "${_prism_jpeg_libs}")
string(REPLACE "\\ " " " _prism_frameworks "${_prism_frameworks}")
file(WRITE "${_env_file}"
"#!/usr/bin/env bash
export AUTOTUNE_DIR=\"${AUTOTUNE_DIR}\"
export AUTOSCHED_BIN=\"${AUTOSCHED_BIN}\"
export HALIDE_DISTRIB_PATH=\"${HALIDE_DIST}\"
export HALIDE_TOOLS_DIR=\"${HALIDE_TOOLS_DIR}\"
export HALIDE_BUILD_DIR=\"${HALIDE_BUILD_DIR}\"
export AUTOTUNE_BIN=\"${AUTOTUNE_BIN}\"
export SAMPLES_DIR=\"${SAMPLES_DIR}\"
export PRISM_PNG_CFLAGS=\"${_prism_png_cflags}\"
export PRISM_PNG_LIBS=\"${_prism_png_libs}\"
export PRISM_JPEG_CFLAGS=\"${_prism_jpeg_cflags}\"
export PRISM_JPEG_LIBS=\"${_prism_jpeg_libs}\"
export PRISM_AUTOTUNE_FRAMEWORKS=\"${_prism_frameworks}\"
${_nvidia_smi_export}
")
file(CHMOD "${_env_file}"
  PERMISSIONS
    OWNER_READ OWNER_WRITE OWNER_EXECUTE
    GROUP_READ GROUP_EXECUTE
    WORLD_READ WORLD_EXECUTE
)
unset(_prism_png_cflags)
unset(_prism_png_libs)
unset(_prism_jpeg_cflags)
unset(_prism_jpeg_libs)
unset(_prism_frameworks)
