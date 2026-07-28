@echo off
chcp 65001 >nul
setlocal EnableExtensions
cd /d "%~dp0"

set "CONDA_ENV=subgen"
set "SUBGEN_URL=http://localhost:8501"
set "HF_HOME=%CD%\.cache\huggingface"
set "CONDA_PKGS_DIRS=%CD%\.conda\pkgs"
set "CONDA_ENVS_PATH=%CD%\.conda\envs"
set "CONDA_NOTICES=false"

echo Starting local Subtitle Generator...
echo URL: %SUBGEN_URL%
echo Model cache: %HF_HOME%
echo.

call :find_python
if not defined PYTHON_EXE (
    echo No ready environment was found.
    echo Run setup.bat first, then start again.
    pause
    exit /b 1
)

start "" powershell -NoProfile -ExecutionPolicy Bypass -Command "Start-Sleep -Seconds 3; Start-Process '%SUBGEN_URL%'"
call "%PYTHON_EXE%" -m subgen.ui --host localhost --port 8501

echo.
echo Subtitle Generator stopped.
pause
exit /b 0

:find_python
set "PYTHON_EXE="
call :find_conda
if not defined CONDA_CMD goto :find_venv_python
call :resolve_conda_activate
if not defined CONDA_ACTIVATE goto :find_venv_python
call "%CONDA_ACTIVATE%" "%CONDA_ENV%" >nul 2>nul
if errorlevel 1 goto :find_venv_python
for /f "delims=" %%p in ('where python 2^>nul') do (
    if not defined PYTHON_EXE set "PYTHON_EXE=%%p"
)
if defined PYTHON_EXE exit /b 0

:find_venv_python
if exist ".venv\Scripts\python.exe" (
    set "PYTHON_EXE=%CD%\.venv\Scripts\python.exe"
    )
exit /b 0

:resolve_conda_activate
set "CONDA_ACTIVATE="
for %%d in (
    "%USERPROFILE%\miniconda3"
    "%USERPROFILE%\anaconda3"
    "%USERPROFILE%\Anaconda3"
    "C:\ProgramData\miniconda3"
    "C:\ProgramData\anaconda3"
    "D:\Anaconda"
) do (
    if exist "%%~d\Scripts\activate.bat" (
        set "CONDA_ACTIVATE=%%~d\Scripts\activate.bat"
        exit /b 0
    )
)
exit /b 0

:find_conda
set "CONDA_CMD="
for /f "delims=" %%c in ('where conda 2^>nul') do (
    if not defined CONDA_CMD set "CONDA_CMD=%%c"
)
if defined CONDA_CMD exit /b 0
for %%d in (
    "%USERPROFILE%\miniconda3"
    "%USERPROFILE%\anaconda3"
    "%USERPROFILE%\Anaconda3"
    "C:\ProgramData\miniconda3"
    "C:\ProgramData\anaconda3"
    "D:\Anaconda"
) do (
    if exist "%%~d\condabin\conda.bat" (
        set "CONDA_CMD=%%~d\condabin\conda.bat"
        exit /b 0
    )
    if exist "%%~d\Scripts\conda.exe" (
        set "CONDA_CMD=%%~d\Scripts\conda.exe"
        exit /b 0
    )
)
exit /b 0
