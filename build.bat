@echo off
setlocal EnableDelayedExpansion

:: Build script for Windows

set "BUILD_TYPE=Release"
set "CUDA_ARCH=80"

:: Parse arguments
:parse_args
if "%~1"=="" goto :done_args
if "%~1"=="--debug" (
    set "BUILD_TYPE=Debug"
    shift
    goto :parse_args
)
if "%~1"=="--arch" (
    set "CUDA_ARCH=%~2"
    shift
    shift
    goto :parse_args
)
if "%~1"=="--clean" (
    if exist build rmdir /s /q build
    shift
    goto :parse_args
)
shift
goto :parse_args
:done_args

echo Building CUDA Operators...
echo Build Type: %BUILD_TYPE%
echo CUDA Arch: sm_%CUDA_ARCH%

:: Create build directory
if not exist build mkdir build
cd build

:: Configure with CMake
echo Configuring CMake...
cmake .. -G "Visual Studio 17 2022" -A x64 ^
    -DCMAKE_CUDA_ARCHITECTURES=%CUDA_ARCH% ^
    -DCMAKE_BUILD_TYPE=%BUILD_TYPE%

if errorlevel 1 (
    echo CMake configuration failed!
    exit /b 1
)

:: Build
echo Building...
cmake --build . --config %BUILD_TYPE% --parallel

if errorlevel 1 (
    echo Build failed!
    exit /b 1
)

echo Build completed successfully!
cd ..

:: Copy built module to project root for testing
if exist build\%BUILD_TYPE%\cuda_ops.cp*.pyd (
    copy /y build\%BUILD_TYPE%\cuda_ops.cp*.pyd . >nul 2>&1
    echo Module copied to project root
)

endlocal
