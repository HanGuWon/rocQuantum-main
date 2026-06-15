@echo off
setlocal

set "PROJECT_DIR=%~dp0"
set "BUILD_DIR=%PROJECT_DIR%build"
set "CMAKE_EXE=cmake"

if defined CMAKE_BIN_PATH (
    set "CMAKE_EXE=%CMAKE_BIN_PATH%\cmake.exe"
)

if defined ROCM_PATH (
    set "PATH=%ROCM_PATH%\bin;%PATH%"
)

if not defined CMAKE_HIP_ARCHITECTURES (
    set "CMAKE_HIP_ARCHITECTURES=gfx950;gfx942;gfx90a"
)

echo Configuring rocQuantum from "%PROJECT_DIR%"
echo Build directory: "%BUILD_DIR%"
echo HIP architectures: %CMAKE_HIP_ARCHITECTURES%

"%CMAKE_EXE%" -S "%PROJECT_DIR%" -B "%BUILD_DIR%" ^
    -DCMAKE_HIP_ARCHITECTURES="%CMAKE_HIP_ARCHITECTURES%"

if %errorlevel% neq 0 (
    echo CMake configuration failed.
    exit /b %errorlevel%
)

"%CMAKE_EXE%" --build "%BUILD_DIR%" --config Release
exit /b %errorlevel%
