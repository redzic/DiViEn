cmake_minimum_required(VERSION 3.26)

# set(CMAKE_C_COMPILER clang)
# set(CMAKE_CXX_COMPILER clang++)

project(DiViEn)


set(CMAKE_CXX_STANDARD 23)
set(CMAKE_BUILD_TYPE Release)
# set(CMAKE_BUILD_TYPE Debug)

set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
set(CMAKE_COLOR_MAKEFILE TRUE)
set(CMAKE_COLOR_DIAGNOSTICS ON)

set(ENABLE_IO_URING OFF CACHE BOOL "Use io-uring for tcp on linux (default off)")
set(CONFIG_MSYS_UCRT64 OFF CACHE BOOL "Enable config for building on msys64 with ucrt64")

option(ENABLE_IO_URING "Use io-uring for tcp on linux (default off)" OFF)
option(CONFIG_MSYS_UCRT64 "Enable config for building on msys64 with ucrt64" OFF)

if(ENABLE_IO_URING)
  add_definitions(-DASIO_HAS_IO_URING=1)
  add_definitions(-DASIO_DISABLE_EPOLL=1)
endif()

if (CONFIG_MSYS_UCRT64)
  add_definitions(-DWINVER=0x0A00)
  add_definitions(-D_WIN32_WINNT=0x0A00)
  set(CMAKE_NO_SYSTEM_FROM_IMPORTED 1)
endif()

# add_compile_options(-fsanitize=undefined -fsanitize=address -fsanitize=bounds -fsanitize=unsigned-integer-overflow  -fsanitize=local-bounds -fsanitize=signed-integer-overflow,null,alignment -fno-sanitize-recover=null -fsanitize-trap=alignment -fno-omit-frame-pointer)
# add_link_options(-fsanitize=undefined -fsanitize=address -fsanitize=bounds -fsanitize=unsigned-integer-overflow  -fsanitize=local-bounds -fsanitize=signed-integer-overflow,null,alignment -fno-sanitize-recover=null -fsanitize-trap=alignment -fno-omit-frame-pointer)

# required for no dependencies on Windows (other than ffmpeg shared libs)

# compile options needed for self-contained binary
# add_compile_options(-static-libstdc++ -static-libgcc -static)
# add_link_options(-static-libstdc++ -static-libgcc -static)

# Turning off exceptions causes the program to abort
# if an exception is thrown.
# add_compile_options(-fno-exceptions -fno-rtti)
# add_link_options(-fno-exceptions -fno-rtti)

find_package(PkgConfig REQUIRED)
pkg_check_modules(LIBAV REQUIRED IMPORTED_TARGET
    libavdevice
    libavfilter
    libavformat
    libavcodec
    libswresample
    libswscale
    libavutil
)


# This is pretty hacky, but whatever.
if(CONFIG_MSYS_UCRT64)
  set_property(TARGET PkgConfig::LIBAV PROPERTY INTERFACE_INCLUDE_DIRECTORIES "/c/msys64/ucrt64/include/")
endif()

# TODO can we isolate the code so that we turn on exceptions
# for only part of the code? like the part that deals with ASIO.
# I think it's possible.

add_executable(DiViEn main.cpp decode.cpp segment.cpp)

target_compile_options(DiViEn PRIVATE -Wall -Wextra -Wformat -Wno-c99-designator )
if(ENABLE_IO_URING)
    target_link_libraries( DiViEn uring )
endif()
target_link_libraries( DiViEn PkgConfig::LIBAV )
