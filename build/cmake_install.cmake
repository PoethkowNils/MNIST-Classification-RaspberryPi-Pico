# Install script for directory: /home/nils03/Uni/ML_ES/IAS0360_lab_excercises_2024/lab_7

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/usr/local")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "TRUE")
endif()

# Set default install directory permissions.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/usr/bin/arm-none-eabi-objdump")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for each subdirectory.
  include("/home/nils03/Uni/ML_ES/IAS0360_lab_excercises_2024/lab_7/build/pico-sdk/cmake_install.cmake")
  include("/home/nils03/Uni/ML_ES/IAS0360_lab_excercises_2024/lab_7/build/src/cmake_install.cmake")
  include("/home/nils03/Uni/ML_ES/IAS0360_lab_excercises_2024/lab_7/build/models/cmake_install.cmake")
  include("/home/nils03/Uni/ML_ES/IAS0360_lab_excercises_2024/lab_7/build/lib/pico-tflmicro/cmake_install.cmake")
  include("/home/nils03/Uni/ML_ES/IAS0360_lab_excercises_2024/lab_7/build/lib/config/cmake_install.cmake")
  include("/home/nils03/Uni/ML_ES/IAS0360_lab_excercises_2024/lab_7/build/lib/lcd/cmake_install.cmake")
  include("/home/nils03/Uni/ML_ES/IAS0360_lab_excercises_2024/lab_7/build/lib/font/cmake_install.cmake")

endif()

if(CMAKE_INSTALL_COMPONENT)
  set(CMAKE_INSTALL_MANIFEST "install_manifest_${CMAKE_INSTALL_COMPONENT}.txt")
else()
  set(CMAKE_INSTALL_MANIFEST "install_manifest.txt")
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
file(WRITE "/home/nils03/Uni/ML_ES/IAS0360_lab_excercises_2024/lab_7/build/${CMAKE_INSTALL_MANIFEST}"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")