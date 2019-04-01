cmake_minimum_required(VERSION 3.4.1)

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/CMakeModules)
PROJECT(server_ffmpeg)

add_definitions("-Wall -g")

find_package(OpenCV REQUIRED)
find_package(SDL REQUIRED)
#find_package(FFmpeg REQUIRED)

if (FFMPEG_FOUND)
  message("Found FFMPEG/LibAV libraries")
  include_directories(${FFMPEG_INCLUDE_DIR})
else (FFMPEG_FOUND)
  message("Can't find libavcodec, libavformat or libavutil. Add them!")
endif (FFMPEG_FOUND)

include_directories(${FFMPEG_INCLUDE_DIR})
SET(SOURCE_FILES streamVideo.cpp)

include_directories(${OpenCV_INCLUCE_DIRS} ${SDL_INCLUDE_DIR})

ADD_EXECUTABLE(${PROJECT_NAME} ${SOURCE_FILES})

TARGET_LINK_LIBRARIES(${PROJECT_NAME} ${OpenCV_LIBS} ${SDL_LIBRARY} ${FFMPEG_LIBRARIES} ${SWSCALE_LIBRARIES} libpthread.so libavformat.so libavutil.so libswscale.so libavcodec.so
#${AVDEVICE_LIBRARIES}
)
