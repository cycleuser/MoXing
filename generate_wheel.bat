@echo off
REM MoXing - Generate wheel package
REM
REM Usage:
REM   generate_wheel.bat              Build wheel with current version
REM   generate_wheel.bat --version 0.2.0   Set specific version
REM
REM Output: dist\moxing-{version}-py3-none-any.whl

setlocal EnableDelayedExpansion
cd /d "%~dp0"

if not defined PYTHON set "PYTHON=python"
set "VERSION_FILE=moxing\__init__.py"

REM Parse arguments
set "VERSION="

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
if /i "%~1"=="--help" goto :show_help
if /i "%~1"=="-h" goto :show_help
echo Unknown option: %~1
exit /b 1

:show_help
echo Usage: generate_wheel.bat [OPTIONS]
echo.
echo Options:
echo   --version, -v VERSION   Set specific version
echo   --help, -h              Show this help
echo.
echo Output: dist\moxing-{version}-py3-none-any.whl
exit /b 0

:done_args

echo === MoXing Wheel Generator ===

REM Set version if specified
if defined VERSION (
    echo [1/4] Setting version to %VERSION%...
    powershell -Command "(Get-Content '%VERSION_FILE%') -replace '__version__ = .*', '__version__ = \"%VERSION%\"' | Set-Content '%VERSION_FILE%'"
) else (
    echo [1/4] Using current version...
)

REM Read current version
for /f "tokens=2 delims==" %%v in ('findstr "__version__" "%VERSION_FILE%"') do (
    for /f "tokens=1 delims= " %%a in ("%%v") do set "CURRENT_VERSION=%%~a"
)
echo   Version: %CURRENT_VERSION%

echo [2/4] Cleaning old builds...
if exist dist\*.whl del /q dist\*.whl
if exist dist\*.tar.gz del /q dist\*.tar.gz
if exist build rmdir /s /q build
for /d %%i in (*.egg-info) do rmdir /s /q "%%i"

echo [3/4] Installing build tools...
%PYTHON% -m pip install --upgrade build -q

echo [4/4] Building wheel...
%PYTHON% scripts\build_platform_wheels.py
if %errorlevel% neq 0 (echo Build failed! & exit /b 1)

echo.
echo === Done! ===
echo Version: %CURRENT_VERSION%
echo Wheel: dist\moxing-%CURRENT_VERSION%-py3-none-any.whl
echo.
echo To upload to PyPI:
echo   upload_pypi.bat

endlocal