# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.24

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
CMAKE_COMMAND = /home/spack/spack/opt/spack/linux-debian12-broadwell/gcc-12.2.0/cmake-3.24.4-64tnq3z4gpajebgx5fd7syzkj5gkx2yj/bin/cmake

# The command to remove a file.
RM = /home/spack/spack/opt/spack/linux-debian12-broadwell/gcc-12.2.0/cmake-3.24.4-64tnq3z4gpajebgx5fd7syzkj5gkx2yj/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/course/hpc/users/2023010828/PA3

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/course/hpc/users/2023010828/PA3/build

# Utility rule file for ContinuousTest.

# Include any custom commands dependencies for this target.
include CMakeFiles/ContinuousTest.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/ContinuousTest.dir/progress.make

CMakeFiles/ContinuousTest:
	/home/spack/spack/opt/spack/linux-debian12-broadwell/gcc-12.2.0/cmake-3.24.4-64tnq3z4gpajebgx5fd7syzkj5gkx2yj/bin/ctest -D ContinuousTest

ContinuousTest: CMakeFiles/ContinuousTest
ContinuousTest: CMakeFiles/ContinuousTest.dir/build.make
.PHONY : ContinuousTest

# Rule to build all files generated by this target.
CMakeFiles/ContinuousTest.dir/build: ContinuousTest
.PHONY : CMakeFiles/ContinuousTest.dir/build

CMakeFiles/ContinuousTest.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/ContinuousTest.dir/cmake_clean.cmake
.PHONY : CMakeFiles/ContinuousTest.dir/clean

CMakeFiles/ContinuousTest.dir/depend:
	cd /home/course/hpc/users/2023010828/PA3/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/course/hpc/users/2023010828/PA3 /home/course/hpc/users/2023010828/PA3 /home/course/hpc/users/2023010828/PA3/build /home/course/hpc/users/2023010828/PA3/build /home/course/hpc/users/2023010828/PA3/build/CMakeFiles/ContinuousTest.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/ContinuousTest.dir/depend

