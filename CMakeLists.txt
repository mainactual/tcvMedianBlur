PROJECT(tcvMedianBlur)

CMAKE_MINIMUM_REQUIRED( VERSION 2.6 )

#
# Open Computer Vision required
#
FIND_PACKAGE(OpenCV)
IF (NOT OpenCV_FOUND )
	MESSAGE(FATAL_ERROR "Package OpenCV not found.")
ENDIF()

SET(SRCS
	main.cpp
	tcvMedianBlur16U.hpp
	tcvMedianBlur16U.cpp
)

ADD_EXECUTABLE(tcvMedianBlur ${SRCS} )	

SET(LIBS
	${OpenCV_LIBRARIES}
)
TARGET_LINK_LIBRARIES(tcvMedianBlur ${LIBS} )
