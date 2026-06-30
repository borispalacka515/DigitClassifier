cmake_minimum_required(VERSION 3.20)

project(DigitClassifier VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(Qt6_VERSION 6.10.1)
set(QT_INSTALL_PATH "D:/Program Files/Qt/${Qt6_VERSION}/msvc2022_64")

set(Qt6_DIR "${QT_INSTALL_PATH}/lib/cmake/Qt6")

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTORCC ON)

find_package(Qt6 REQUIRED COMPONENTS Widgets Core Gui)

include(FetchContent)

FetchContent_Declare(
    nlohmann_json
    URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
)

FetchContent_MakeAvailable(nlohmann_json)

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
    src/core/ModelConfig.cpp
    src/core/Model.h
    src/core/Model.cpp
    src/core/ModelSerializer.h
    src/core/ModelSerializer.cpp
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
    src/widgets/DatasetViewerPage.h
    src/widgets/DatasetViewerPage.cpp
    src/widgets/DatasetViewerPage.ui
    src/Widgets/PredictionBarChartWidget.h
    src/Widgets/PredictionBarChartWidget.cpp
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

set(TRAINING_SOURCES
    src/training/Trainer.h
    src/training/Trainer.cpp
    src/training/TrainingConfig.h
    src/training/TrainingConfig.cpp
    src/training/LossFunction.h
    src/training/LossFunction.cpp
)

set(PROJECT_SOURCES
    ${APP_SOURCES}
    ${CORE_SOURCES}
    ${DATA_SOURCES}
    ${IMAGE_SOURCES}
    ${WIDGET_SOURCES}
    ${GALLERY_SOURCES}
    ${DIALOG_SOURCES}
    ${TRAINING_SOURCES}
)

qt_add_executable(${PROJECT_NAME}
    ${PROJECT_SOURCES}
)

target_link_libraries(${PROJECT_NAME}
    PRIVATE
        Qt6::Widgets
        Qt6::Core
        Qt6::Gui
        nlohmann_json::nlohmann_json
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
        src/training
)

source_group("DigitClassifier\\App" FILES ${APP_SOURCES})
source_group("DigitClassifier\\Core" FILES ${CORE_SOURCES})
source_group("DigitClassifier\\Data" FILES ${DATA_SOURCES})
source_group("DigitClassifier\\Image" FILES ${IMAGE_SOURCES})
source_group("DigitClassifier\\Widgets" FILES ${WIDGET_SOURCES})
source_group("DigitClassifier\\Gallery" FILES ${GALLERY_SOURCES})
source_group("DigitClassifier\\Dialog" FILES ${DIALOG_SOURCES})
source_group("DigitClassifier\\Training" FILES ${TRAINING_SOURCES})

add_custom_command(TARGET ${PROJECT_NAME}
    POST_BUILD
    COMMAND "${QT_INSTALL_PATH}/bin/windeployqt.exe" "$<TARGET_FILE:${PROJECT_NAME}>"
    COMMENT "Deploying Qt runtime dependencies with windeployqt"
)

set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    PROPERTY VS_STARTUP_PROJECT ${PROJECT_NAME}
)