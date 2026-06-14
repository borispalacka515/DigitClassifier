cmake_minimum_required(VERSION 3.20)

project(DigitClassifier VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTORCC ON)

find_package(Qt6 REQUIRED COMPONENTS Widgets Core Gui)

set(APP_SOURCES
    src/app/main.cpp
    src/app/DigitClassifier.h
    src/app/DigitClassifier.cpp
    src/app/DigitClassifier.ui
    src/app/DigitClassifier.qrc
)

set(CORE_SOURCES
    src/core/Activation.h
    src/core/Activation.cpp
    src/core/DenseLayer.h
    src/core/DenseLayer.cpp
    src/core/ModelConfig.h
    src/core/Model.h
    src/core/Model.cpp
)

set(DATA_SOURCES
    src/data/Dataset.h
    src/data/Dataset.cpp
    src/data/MNIST.h
    src/data/MNISTLoader.h
    src/data/MNISTLoader.cpp
    src/data/DatasetLoadWorker.h
    src/data/DatasetLoadWorker.cpp
)

set(IMAGE_SOURCES
    src/image/ImageConverter.h
    src/image/ImageConverter.cpp
)

set(WIDGET_SOURCES
    src/widgets/SampleViewerWidget.h
    src/widgets/SampleViewerWidget.cpp
)

set(GALLERY_SOURCES
    src/gallery/SampleThumbnailModel.h
    src/gallery/SampleThumbnailModel.cpp
)

set(DIALOG_SOURCES
    src/dialog/DatasetOpenDialog.h
    src/dialog/DatasetOpenDialog.cpp
    src/dialog/DatasetOpenDialog.ui
)

set(PROJECT_SOURCES
    ${APP_SOURCES}
    ${CORE_SOURCES}
    ${DATA_SOURCES}
    ${IMAGE_SOURCES}
    ${WIDGET_SOURCES}
    ${GALLERY_SOURCES}
    ${DIALOG_SOURCES}
)

qt_add_executable(${PROJECT_NAME}
    ${PROJECT_SOURCES}
)

target_link_libraries(${PROJECT_NAME}
    PRIVATE
        Qt6::Widgets
        Qt6::Core
        Qt6::Gui
)

target_include_directories(${PROJECT_NAME}
    PRIVATE
        src
        src/app
        src/core
        src/data
        src/image
        src/widgets
        src/gallery
        src/dialog
)

source_group("DigitClassifier\\App" FILES ${APP_SOURCES})
source_group("DigitClassifier\\Core" FILES ${CORE_SOURCES})
source_group("DigitClassifier\\Data" FILES ${DATA_SOURCES})
source_group("DigitClassifier\\Image" FILES ${IMAGE_SOURCES})
source_group("DigitClassifier\\Widgets" FILES ${WIDGET_SOURCES})
source_group("DigitClassifier\\Gallery" FILES ${GALLERY_SOURCES})
source_group("DigitClassifier\\Dialog" FILES ${DIALOG_SOURCES})

if(WIN32)
    find_program(WINDEPLOYQT_EXECUTABLE windeployqt HINTS "${Qt6_DIR}/../../../bin")

    if(WINDEPLOYQT_EXECUTABLE)
        add_custom_command(TARGET ${PROJECT_NAME}
            POST_BUILD
            COMMAND "${WINDEPLOYQT_EXECUTABLE}" "$<TARGET_FILE:${PROJECT_NAME}>"
            COMMENT "Deploying Qt runtime dependencies with windeployqt"
        )
    endif()
endif()

set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    PROPERTY VS_STARTUP_PROJECT ${PROJECT_NAME}
)