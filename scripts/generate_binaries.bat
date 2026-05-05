@echo off
REM MoXing - Build llama.cpp binaries from source
REM
REM This script builds llama.cpp binaries from source for multiple backends.
REM Each backend requires specific dependencies:
REM   - CPU: cmake, Visual Studio (C++ workload)
REM   - CUDA: CUDA Toolkit (nvcc)
REM   - Vulkan: Vulkan SDK
REM
REM Usage:
REM   generate_binaries.bat                    Build all available backends
REM   generate_binaries.bat --backend cuda     Build specific backend
REM   generate_binaries.bat --version b8468    Build specific version
REM   generate_binaries.bat --clean            Clean build directory first
REM
REM Output: moxing\bin\{platform}-{backend}\
REM         dist\binaries\{platform}-{backend}.zip

setlocal EnableDelayedExpansion
cd /d "%~dp0"

set "SCRIPT_DIR=%~dp0"
set "PYTHON=python"
set "BIN_DIR=%SCRIPT_DIR%moxing\bin"
set "DIST_DIR=%SCRIPT_DIR%dist\binaries"
set "BUILD_DIR=%SCRIPT_DIR%build\llama.cpp"
set "LLAMA_CPP_REPO=ggml-org/llama.cpp"

REM Detect platform
set "OS=windows"
set "ARCH=x64"
if "%PROCESSOR_ARCHITECTURE%"=="ARM64" set "ARCH=arm64"
set "PLATFORM=%OS%-%ARCH%"

REM Default values
set "VERSION="
set "BACKENDS=all"
set "CLEAN=0"
set "JOBS=%NUMBER_OF_PROCESSORS%"

REM Parse arguments
:parse_args
if "%~1"=="" goto :done_args
if /i "%~1"=="--version" (
    set "VERSION=%~2"
    shift
    shift
    goto :parse_args
)
if /i "%~1"=="-v" (
    set "VERSION=%~2"
    shift
    shift
    goto :parse_args
)
if /i "%~1"=="--backend" (
    set "BACKENDS=%~2"
    shift
    shift
    goto :parse_args
)
if /i "%~1"=="-b" (
    set "BACKENDS=%~2"
    shift
    shift
    goto :parse_args
)
if /i "%~1"=="--clean" (
    set "CLEAN=1"
    shift
    goto :parse_args
)
if /i "%~1"=="-c" (
    set "CLEAN=1"
    shift
    goto :parse_args
)
if /i "%~1"=="--jobs" (
    set "JOBS=%~2"
    shift
    shift
    goto :parse_args
)
if /i "%~1"=="-j" (
    set "JOBS=%~2"
    shift
    shift
    goto :parse_args
)
if /i "%~1"=="--help" goto :show_help
if /i "%~1"=="-h" goto :show_help
echo Unknown option: %~1
exit /b 1

:show_help
echo MoXing - Build llama.cpp binaries from source
echo.
echo Usage: generate_binaries.bat [OPTIONS]
echo.
echo Options:
echo   --version, -v VERSION    Build specific llama.cpp version (tag)
echo   --backend, -b BACKEND    Build specific backend (cpu,cuda,vulkan,all)
echo   --clean, -c              Clean build directory first
echo   --jobs, -j N             Number of parallel jobs (default: %JOBS%)
echo   --help, -h               Show this help
echo.
echo Backends:
echo   cpu     - CPU only (requires: cmake, Visual Studio)
echo   cuda    - NVIDIA CUDA (requires: CUDA Toolkit)
echo   vulkan  - Vulkan (requires: Vulkan SDK)
echo   all     - Build all available backends
echo.
echo Output:
echo   moxing\bin\{platform}-{backend}\
echo   dist\binaries\{platform}-{backend}.zip
echo.
echo Examples:
echo   generate_binaries.bat --backend cuda
echo   generate_binaries.bat --version b8468 --clean
exit /b 0

:done_args

echo === MoXing Binary Builder ===
echo.
echo Platform: %PLATFORM%
echo Jobs: %JOBS%

REM Check for required tools
where cmake >nul 2>nul
if %errorlevel% neq 0 (
    echo Error: cmake not found. Please install cmake and add to PATH.
    exit /b 1
)

where git >nul 2>nul
if %errorlevel% neq 0 (
    echo Error: git not found. Please install git and add to PATH.
    exit /b 1
)

REM Get latest version if not specified
if not defined VERSION (
    echo [1/5] Fetching latest llama.cpp version...
    for /f "delims=" %%i in ('%PYTHON% -c "import json;from urllib.request import urlopen,Request;r=urlopen(Request('https://api.github.com/repos/%LLAMA_CPP_REPO%/releases/latest',headers={'Accept':'application/vnd.github.v3+json','User-Agent':'moxing'}));print(json.loads(r.read().decode())['tag_name'])"') do set "VERSION=%%i"
    echo   Latest version: %VERSION%
) else (
    echo [1/5] Using specified version: %VERSION%
)

REM Clone or update llama.cpp
echo [2/5] Preparing llama.cpp source...
if "%CLEAN%"=="1" (
    if exist "%BUILD_DIR%" (
        echo   Cleaning build directory...
        rmdir /s /q "%BUILD_DIR%"
    )
)

if not exist "%BUILD_DIR%" (
    echo   Cloning llama.cpp...
    git clone --depth 1 --branch "%VERSION%" "https://github.com/%LLAMA_CPP_REPO%.git" "%BUILD_DIR%"
) else (
    echo   Updating llama.cpp...
    cd /d "%BUILD_DIR%"
    git fetch --tags
    git checkout "%VERSION%" 2>nul || git checkout "tags/%VERSION%"
    git submodule update --init --recursive
    cd /d "%SCRIPT_DIR%"
)

REM Create output directories
if not exist "%BIN_DIR%" mkdir "%BIN_DIR%"
if not exist "%DIST_DIR%" mkdir "%DIST_DIR%"

REM Build function
goto :build_main

:build_backend
setlocal
set "BACKEND=%~1"
set "PLATFORM_NAME=%PLATFORM%-%BACKEND%"
set "OUTPUT_DIR=%BIN_DIR%\%PLATFORM_NAME%"
set "LLAMA_BUILD_DIR=%BUILD_DIR%\build-%BACKEND%"

echo.
echo === Building: %PLATFORM_NAME% ===

REM Clean previous build
if exist "%LLAMA_BUILD_DIR%" rmdir /s /q "%LLAMA_BUILD_DIR%"
if exist "%OUTPUT_DIR%" rmdir /s /q "%OUTPUT_DIR%"
mkdir "%OUTPUT_DIR%"

cd /d "%BUILD_DIR%"

REM Set cmake options
set "CMAKE_OPTS=-DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=ON -DLLAMA_BUILD_SERVER=ON"

if "%BACKEND%"=="cpu" (
    set "CMAKE_OPTS=%CMAKE_OPTS% -DLLAMA_CUDA=OFF -DLLAMA_VULKAN=OFF"
    goto :do_build
)

if "%BACKEND%"=="cuda" (
    where nvcc >nul 2>nul
    if !errorlevel! neq 0 (
        echo   Skipping: CUDA not available (nvcc not found)
        exit /b 0
    )
    set "CMAKE_OPTS=%CMAKE_OPTS% -DLLAMA_CUDA=ON -DLLAMA_VULKAN=OFF"
    goto :do_build
)

if "%BACKEND%"=="vulkan" (
    if not defined VULKAN_SDK (
        echo   Skipping: Vulkan not available (VULKAN_SDK not set)
        exit /b 0
    )
    set "CMAKE_OPTS=%CMAKE_OPTS% -DLLAMA_CUDA=OFF -DLLAMA_VULKAN=ON"
    goto :do_build
)

echo   Unknown backend: %BACKEND%
exit /b 1

:do_build
echo   CMake options: %CMAKE_OPTS%

REM Configure
cmake -B "%LLAMA_BUILD_DIR%" -S . %CMAKE_OPTS% -A x64
if !errorlevel! neq 0 (
    echo   CMake configuration failed!
    exit /b 1
)

REM Build
cmake --build "%LLAMA_BUILD_DIR%" --config Release -j %JOBS%
if !errorlevel! neq 0 (
    echo   Build failed!
    exit /b 1
)

REM Copy binaries
echo   Copying binaries...
set "BIN_SRC=%LLAMA_BUILD_DIR%\bin\Release"
if not exist "!BIN_SRC!" set "BIN_SRC=%LLAMA_BUILD_DIR%\bin"
if not exist "!BIN_SRC!" set "BIN_SRC=%LLAMA_BUILD_DIR%\Release"

if exist "!BIN_SRC!" (
    for %%f in ("!BIN_SRC!\llama-*.exe") do (
        copy "%%f" "%OUTPUT_DIR%\" >nul
        echo     %%~nxf
    )
    for %%f in ("!BIN_SRC!\*.dll") do (
        copy "%%f" "%OUTPUT_DIR%\" >nul
        echo     %%~nxf
    )
)

REM Create VERSION file
echo %VERSION%> "%OUTPUT_DIR%\VERSION"
echo %BACKEND%>> "%OUTPUT_DIR%\VERSION"

REM Count files
for /f %%i in ('dir /b "%OUTPUT_DIR%" 2^>nul ^| find /c /v ""') do set "COUNT=%%i"
echo   Done: %COUNT% files

cd /d "%SCRIPT_DIR%"
endlocal
exit /b 0

:build_main
echo [3/5] Building binaries...

if "%BACKENDS%"=="all" (
    call :build_backend "cpu"
    call :build_backend "cuda"
    call :build_backend "vulkan"
) else (
    for %%b in (%BACKENDS%) do call :build_backend "%%b"
)

REM Package binaries
echo.
echo [4/5] Packaging binaries...
%PYTHON% scripts\upload_binaries.py --package

REM Summary
echo.
echo [5/5] Summary...
echo.
echo Binaries built to: moxing\bin\
dir /b "%BIN_DIR%\*\VERSION" 2>nul || echo   No binaries built
echo.
echo Packages created in: dist\binaries\
dir /b "%DIST_DIR%\*.zip" 2>nul || echo   No packages created

echo.
echo === Done! ===
echo.
echo llama.cpp version: %VERSION%
echo Platform: %PLATFORM%
echo.
echo To upload binaries to GitHub:
echo   upload_binaries.bat

endlocal