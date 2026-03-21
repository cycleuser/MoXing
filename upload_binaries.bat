@echo off
REM MoXing - Upload binaries to GitHub releases
REM
REM This script uploads the packaged binaries to GitHub releases.
REM Run generate_binaries.bat first to download and package the binaries.
REM
REM Usage:
REM   upload_binaries.bat              Upload to GitHub
REM   upload_binaries.bat --package    Package only, don't upload
REM
REM Prerequisites:
REM   gh auth login
REM
REM Release URL: https://github.com/cycleuser/MoXing/releases/tag/binaries

setlocal EnableDelayedExpansion
cd /d "%~dp0"

if not defined PYTHON set "PYTHON=python"
set "MOXING_REPO=cycleuser/MoXing"
set "RELEASE_TAG=binaries"
set "DIST_DIR=dist\binaries"

REM Parse arguments
set "PACKAGE_ONLY=0"

:parse_args
if "%~1"=="" goto :done_args
if /i "%~1"=="--package" (
    set "PACKAGE_ONLY=1"
    shift
    goto :parse_args
)
if /i "%~1"=="-p" (
    set "PACKAGE_ONLY=1"
    shift
    goto :parse_args
)
if /i "%~1"=="--help" goto :show_help
if /i "%~1"=="-h" goto :show_help
echo Unknown option: %~1
exit /b 1

:show_help
echo Usage: upload_binaries.bat [OPTIONS]
echo.
echo Options:
echo   --package, -p    Package binaries only, don't upload
echo   --help, -h       Show this help
echo.
echo Prerequisites:
echo   gh auth login
echo.
echo Release URL: https://github.com/%MOXING_REPO%/releases/tag/%RELEASE_TAG%
exit /b 0

:done_args

echo === MoXing Binary Uploader ===

REM Check if binaries exist
if not exist "%DIST_DIR%" (
    echo Error: No packaged binaries found.
    echo Run generate_binaries.bat first.
    exit /b 1
)

REM Package binaries
echo [1/3] Packaging binaries...
%PYTHON% scripts\upload_binaries.py --package

if "%PACKAGE_ONLY%"=="1" (
    echo.
    echo === Done (package only) ===
    echo Packages in: %DIST_DIR%
    dir /b "%DIST_DIR%"
    exit /b 0
)

REM Check for gh CLI
echo [2/3] Checking GitHub CLI...
where gh >nul 2>nul
if %errorlevel% neq 0 (
    echo Error: GitHub CLI (gh) not installed.
    echo Install from: https://cli.github.com/
    echo.
    echo Then run: gh auth login
    exit /b 1
)

gh auth status >nul 2>nul
if %errorlevel% neq 0 (
    echo Error: Not authenticated with GitHub.
    echo Run: gh auth login
    exit /b 1
)

echo   GitHub CLI is ready

REM Upload to GitHub
echo [3/3] Uploading to GitHub...

REM Get moxing version
for /f "tokens=2 delims==" %%v in ('findstr "__version__" moxing\__init__.py') do (
    for /f "tokens=1 delims= " %%a in ("%%v") do set "MOXING_VERSION=%%~a"
)

REM Get llama.cpp version
set "LLAMA_VERSION=unknown"
for /f "delims=" %%i in ('dir /b moxing\bin\*\VERSION 2^>nul') do (
    for /f "delims=" %%j in ('type "moxing\bin\%%i" 2^>nul ^| findstr /r "^"') do (
        set "LLAMA_VERSION=%%j"
        goto :got_version
    )
)
:got_version

REM Create release notes
set "RELEASE_NOTES=llama.cpp: %LLAMA_VERSION%moxing: %MOXING_VERSION%Pre-built llama.cpp binaries for MoXing.## Supported Platforms| Platform | CPU | CUDA | Vulkan | ROCm | Metal ||----------|-----|------|--------|------|-------|| Linux x64 | check | check | check | check | - || Windows x64 | check | check | check | - | - || macOS ARM64 | check | - | - | - | check |## Installationpip install moxingmoxing serve model.gguf  # Binaries download automatically"

REM Check if release exists
gh release view "%RELEASE_TAG%" --repo "%MOXING_REPO%" >nul 2>nul
if %errorlevel% equ 0 (
    echo   Release %RELEASE_TAG% exists, updating...
) else (
    echo   Creating release %RELEASE_TAG%...
    gh release create "%RELEASE_TAG%" --repo "%MOXING_REPO%" --title "Binaries" --notes "llama.cpp: %LLAMA_VERSION%"
)

REM Upload assets
for %%f in ("%DIST_DIR%\*") do (
    echo.
    echo   Uploading: %%~nxf
    gh release upload "%RELEASE_TAG%" "%%f" --repo "%MOXING_REPO%" --clobber
    if !errorlevel! neq 0 echo     [warning] Failed to upload %%~nxf
)

echo.
echo === Done! ===
echo.
echo Release URL: https://github.com/%MOXING_REPO%/releases/tag/%RELEASE_TAG%
echo.
echo llama.cpp version: %LLAMA_VERSION%
echo moxing version: %MOXING_VERSION%

endlocal