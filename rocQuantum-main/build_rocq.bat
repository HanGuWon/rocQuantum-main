@echo off
setlocal

REM build_rocq.bat - Windows helper for configuring and building rocQuantum.
REM Override these values in the environment before running the script when needed.

if not defined ROCM_PATH (
    set "ROCM_PATH=C:\Program Files\AMD\ROCm\6.1"
)

if not defined CMAKE_HIP_ARCHITECTURES (
    set "CMAKE_HIP_ARCHITECTURES=gfx950;gfx942;gfx90a"
)

set "PROJECT_DIR=%~dp0"
set "BUILD_DIR=%PROJECT_DIR%build"
set "INSTALL_DIR=%PROJECT_DIR%install"
set "CMAKE_EXE=cmake"

if defined CMAKE_BIN_PATH (
    set "CMAKE_EXE=%CMAKE_BIN_PATH%\cmake.exe"
)

echo ==================================================================
echo                rocQuantum Build Script
echo ==================================================================

if not exist "%ROCM_PATH%" (
    echo Error: ROCM_PATH not found at "%ROCM_PATH%".
    echo Set ROCM_PATH to your ROCm installation directory.
    exit /b 1
)

echo ROCm path: %ROCM_PATH%
echo HIP architectures: %CMAKE_HIP_ARCHITECTURES%

set "PATH=%ROCM_PATH%\bin;%PATH%"
set "HIP_DEVICE_LIB_PATH=%ROCM_PATH%\lib"

if not exist "%BUILD_DIR%" (
    mkdir "%BUILD_DIR%"
)

echo.
echo --- Running CMake ---
echo.
"%CMAKE_EXE%" -S "%PROJECT_DIR%" -B "%BUILD_DIR%" -G "Visual Studio 17 2022" -A x64 ^
    -DCMAKE_BUILD_TYPE=Release ^
    -DCMAKE_CXX_COMPILER="%ROCM_PATH%\bin\hipcc.bat" ^
    -DCMAKE_INSTALL_PREFIX="%INSTALL_DIR%" ^
    -DCMAKE_HIP_ARCHITECTURES="%CMAKE_HIP_ARCHITECTURES%"

if %errorlevel% neq 0 (
    echo CMake configuration failed.
    exit /b %errorlevel%
)

echo.
echo --- Building Project ---
echo.
"%CMAKE_EXE%" --build "%BUILD_DIR%" --config Release

if %errorlevel% neq 0 (
    echo Build failed.
    exit /b %errorlevel%
)

echo.
echo ==================================================================
echo                  Build completed successfully.
echo ==================================================================
exit /b 0
