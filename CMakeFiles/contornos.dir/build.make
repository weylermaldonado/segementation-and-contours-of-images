# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.5

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
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
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/weyler/Documentos/Vision/Contornos

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/weyler/Documentos/Vision/Contornos

# Include any dependencies generated for this target.
include CMakeFiles/contornos.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/contornos.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/contornos.dir/flags.make

CMakeFiles/contornos.dir/contornos.cpp.o: CMakeFiles/contornos.dir/flags.make
CMakeFiles/contornos.dir/contornos.cpp.o: contornos.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/weyler/Documentos/Vision/Contornos/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/contornos.dir/contornos.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/contornos.dir/contornos.cpp.o -c /home/weyler/Documentos/Vision/Contornos/contornos.cpp

CMakeFiles/contornos.dir/contornos.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/contornos.dir/contornos.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/weyler/Documentos/Vision/Contornos/contornos.cpp > CMakeFiles/contornos.dir/contornos.cpp.i

CMakeFiles/contornos.dir/contornos.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/contornos.dir/contornos.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/weyler/Documentos/Vision/Contornos/contornos.cpp -o CMakeFiles/contornos.dir/contornos.cpp.s

CMakeFiles/contornos.dir/contornos.cpp.o.requires:

.PHONY : CMakeFiles/contornos.dir/contornos.cpp.o.requires

CMakeFiles/contornos.dir/contornos.cpp.o.provides: CMakeFiles/contornos.dir/contornos.cpp.o.requires
	$(MAKE) -f CMakeFiles/contornos.dir/build.make CMakeFiles/contornos.dir/contornos.cpp.o.provides.build
.PHONY : CMakeFiles/contornos.dir/contornos.cpp.o.provides

CMakeFiles/contornos.dir/contornos.cpp.o.provides.build: CMakeFiles/contornos.dir/contornos.cpp.o


# Object files for target contornos
contornos_OBJECTS = \
"CMakeFiles/contornos.dir/contornos.cpp.o"

# External object files for target contornos
contornos_EXTERNAL_OBJECTS =

contornos: CMakeFiles/contornos.dir/contornos.cpp.o
contornos: CMakeFiles/contornos.dir/build.make
contornos: /usr/local/lib/libopencv_stitching.so.3.3.1
contornos: /usr/local/lib/libopencv_superres.so.3.3.1
contornos: /usr/local/lib/libopencv_videostab.so.3.3.1
contornos: /usr/local/lib/libopencv_aruco.so.3.3.1
contornos: /usr/local/lib/libopencv_bgsegm.so.3.3.1
contornos: /usr/local/lib/libopencv_bioinspired.so.3.3.1
contornos: /usr/local/lib/libopencv_ccalib.so.3.3.1
contornos: /usr/local/lib/libopencv_cvv.so.3.3.1
contornos: /usr/local/lib/libopencv_dpm.so.3.3.1
contornos: /usr/local/lib/libopencv_face.so.3.3.1
contornos: /usr/local/lib/libopencv_freetype.so.3.3.1
contornos: /usr/local/lib/libopencv_fuzzy.so.3.3.1
contornos: /usr/local/lib/libopencv_img_hash.so.3.3.1
contornos: /usr/local/lib/libopencv_line_descriptor.so.3.3.1
contornos: /usr/local/lib/libopencv_optflow.so.3.3.1
contornos: /usr/local/lib/libopencv_reg.so.3.3.1
contornos: /usr/local/lib/libopencv_rgbd.so.3.3.1
contornos: /usr/local/lib/libopencv_saliency.so.3.3.1
contornos: /usr/local/lib/libopencv_stereo.so.3.3.1
contornos: /usr/local/lib/libopencv_structured_light.so.3.3.1
contornos: /usr/local/lib/libopencv_surface_matching.so.3.3.1
contornos: /usr/local/lib/libopencv_tracking.so.3.3.1
contornos: /usr/local/lib/libopencv_xfeatures2d.so.3.3.1
contornos: /usr/local/lib/libopencv_ximgproc.so.3.3.1
contornos: /usr/local/lib/libopencv_xobjdetect.so.3.3.1
contornos: /usr/local/lib/libopencv_xphoto.so.3.3.1
contornos: /usr/local/lib/libopencv_shape.so.3.3.1
contornos: /usr/local/lib/libopencv_photo.so.3.3.1
contornos: /usr/local/lib/libopencv_calib3d.so.3.3.1
contornos: /usr/local/lib/libopencv_phase_unwrapping.so.3.3.1
contornos: /usr/local/lib/libopencv_video.so.3.3.1
contornos: /usr/local/lib/libopencv_datasets.so.3.3.1
contornos: /usr/local/lib/libopencv_plot.so.3.3.1
contornos: /usr/local/lib/libopencv_text.so.3.3.1
contornos: /usr/local/lib/libopencv_dnn.so.3.3.1
contornos: /usr/local/lib/libopencv_features2d.so.3.3.1
contornos: /usr/local/lib/libopencv_flann.so.3.3.1
contornos: /usr/local/lib/libopencv_highgui.so.3.3.1
contornos: /usr/local/lib/libopencv_ml.so.3.3.1
contornos: /usr/local/lib/libopencv_videoio.so.3.3.1
contornos: /usr/local/lib/libopencv_imgcodecs.so.3.3.1
contornos: /usr/local/lib/libopencv_objdetect.so.3.3.1
contornos: /usr/local/lib/libopencv_imgproc.so.3.3.1
contornos: /usr/local/lib/libopencv_core.so.3.3.1
contornos: CMakeFiles/contornos.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/weyler/Documentos/Vision/Contornos/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable contornos"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/contornos.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/contornos.dir/build: contornos

.PHONY : CMakeFiles/contornos.dir/build

CMakeFiles/contornos.dir/requires: CMakeFiles/contornos.dir/contornos.cpp.o.requires

.PHONY : CMakeFiles/contornos.dir/requires

CMakeFiles/contornos.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/contornos.dir/cmake_clean.cmake
.PHONY : CMakeFiles/contornos.dir/clean

CMakeFiles/contornos.dir/depend:
	cd /home/weyler/Documentos/Vision/Contornos && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/weyler/Documentos/Vision/Contornos /home/weyler/Documentos/Vision/Contornos /home/weyler/Documentos/Vision/Contornos /home/weyler/Documentos/Vision/Contornos /home/weyler/Documentos/Vision/Contornos/CMakeFiles/contornos.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/contornos.dir/depend
