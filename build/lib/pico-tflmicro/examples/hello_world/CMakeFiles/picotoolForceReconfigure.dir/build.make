# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/nils03/Uni/ML_ES/IAS0360_lab_excercises_2024/lab_7

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/nils03/Uni/ML_ES/IAS0360_lab_excercises_2024/lab_7/build

# Utility rule file for picotoolForceReconfigure.

# Include any custom commands dependencies for this target.
include lib/pico-tflmicro/examples/hello_world/CMakeFiles/picotoolForceReconfigure.dir/compiler_depend.make

# Include the progress variables for this target.
include lib/pico-tflmicro/examples/hello_world/CMakeFiles/picotoolForceReconfigure.dir/progress.make

lib/pico-tflmicro/examples/hello_world/CMakeFiles/picotoolForceReconfigure:
	cd /home/nils03/Uni/ML_ES/IAS0360_lab_excercises_2024/lab_7/build/lib/pico-tflmicro/examples/hello_world && /usr/bin/cmake -E touch_nocreate /home/nils03/Uni/ML_ES/IAS0360_lab_excercises_2024/lab_7/CMakeLists.txt

picotoolForceReconfigure: lib/pico-tflmicro/examples/hello_world/CMakeFiles/picotoolForceReconfigure
picotoolForceReconfigure: lib/pico-tflmicro/examples/hello_world/CMakeFiles/picotoolForceReconfigure.dir/build.make
.PHONY : picotoolForceReconfigure

# Rule to build all files generated by this target.
lib/pico-tflmicro/examples/hello_world/CMakeFiles/picotoolForceReconfigure.dir/build: picotoolForceReconfigure
.PHONY : lib/pico-tflmicro/examples/hello_world/CMakeFiles/picotoolForceReconfigure.dir/build

lib/pico-tflmicro/examples/hello_world/CMakeFiles/picotoolForceReconfigure.dir/clean:
	cd /home/nils03/Uni/ML_ES/IAS0360_lab_excercises_2024/lab_7/build/lib/pico-tflmicro/examples/hello_world && $(CMAKE_COMMAND) -P CMakeFiles/picotoolForceReconfigure.dir/cmake_clean.cmake
.PHONY : lib/pico-tflmicro/examples/hello_world/CMakeFiles/picotoolForceReconfigure.dir/clean

lib/pico-tflmicro/examples/hello_world/CMakeFiles/picotoolForceReconfigure.dir/depend:
	cd /home/nils03/Uni/ML_ES/IAS0360_lab_excercises_2024/lab_7/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/nils03/Uni/ML_ES/IAS0360_lab_excercises_2024/lab_7 /home/nils03/Uni/ML_ES/IAS0360_lab_excercises_2024/lab_7/lib/pico-tflmicro/examples/hello_world /home/nils03/Uni/ML_ES/IAS0360_lab_excercises_2024/lab_7/build /home/nils03/Uni/ML_ES/IAS0360_lab_excercises_2024/lab_7/build/lib/pico-tflmicro/examples/hello_world /home/nils03/Uni/ML_ES/IAS0360_lab_excercises_2024/lab_7/build/lib/pico-tflmicro/examples/hello_world/CMakeFiles/picotoolForceReconfigure.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : lib/pico-tflmicro/examples/hello_world/CMakeFiles/picotoolForceReconfigure.dir/depend
