@echo off
REM MoXing - Upload wheel to PyPI
REM
REM This script uploads the built wheel to PyPI.
REM Run generate_wheel.bat first to build the wheel.
REM
REM Usage:
REM   upload_pypi.bat              Upload to PyPI
REM   upload_pypi.bat --test       Upload to TestPyPI
REM   upload_pypi.bat --check      Check wheel only, don't upload
REM
REM Prerequisites:
REM   pip install twine
REM   Set TWINE_USERNAME and TWINE_PASSWORD (or use API token)

setlocal EnableDelayedExpansion
cd /d "%~dp0"

if not defined PYTHON set "PYTHON=python"

REM Parse arguments
set "TEST=0"
set "CHECK_ONLY=0"

:parse_args
if "%~1"=="" goto :done_args
if /i "%~1"=="--test" (
    set "TEST=1"
    shift
    goto :parse_args
)
if /i "%~1"=="-t" (
    set "TEST=1"
    shift
    goto :parse_args
)
if /i "%~1"=="--check" (
    set "CHECK_ONLY=1"
    shift
    goto :parse_args
)
if /i "%~1"=="-c" (
    set "CHECK_ONLY=1"
    shift
    goto :parse_args
)
if /i "%~1"=="--help" goto :show_help
if /i "%~1"=="-h" goto :show_help
echo Unknown option: %~1
exit /b 1

:show_help
echo Usage: upload_pypi.bat [OPTIONS]
echo.
echo Options:
echo   --test, -t    Upload to TestPyPI instead of PyPI
echo   --check, -c   Check wheel only, don't upload
echo   --help, -h    Show this help
echo.
echo Prerequisites:
echo   pip install twine
echo   Set TWINE_USERNAME and TWINE_PASSWORD (or use API token)
exit /b 0

:done_args

echo === MoXing PyPI Uploader ===

REM Find wheel
set "WHEEL="
for /f "delims=" %%i in ('dir /b /o-d dist\*.whl 2^>nul') do (
    set "WHEEL=dist\%%i"
    goto :found_wheel
)

echo Error: No wheel found in dist\
echo Run generate_wheel.bat first.
exit /b 1

:found_wheel
echo Wheel: %WHEEL%

REM Get version from wheel name
for /f "tokens=2 delims=-" %%v in ("%WHEEL%") do set "VERSION=%%v"
echo Version: %VERSION%

REM Check wheel
echo.
echo [1/2] Checking wheel...
%PYTHON% -m pip install --upgrade twine -q
%PYTHON% -m twine check "%WHEEL%"
if %errorlevel% neq 0 (echo Check failed! & exit /b 1)

if "%CHECK_ONLY%"=="1" (
    echo.
    echo === Done (check only) ===
    exit /b 0
)

REM Upload
echo.
echo [2/2] Uploading...

if "%TEST%"=="1" (
    echo Uploading to TestPyPI...
    %PYTHON% -m twine upload --repository testpypi "%WHEEL%"
    echo.
    echo === Done! ===
    echo TestPyPI URL: https://test.pypi.org/project/moxing/
) else (
    echo Uploading to PyPI...
    %PYTHON% -m twine upload "%WHEEL%"
    echo.
    echo === Done! ===
    echo PyPI URL: https://pypi.org/project/moxing/
)

echo.
echo Version: %VERSION%
echo Wheel: %WHEEL%

endlocal