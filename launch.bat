@echo off
chcp 65001 >nul
setlocal EnableExtensions
cd /d "%~dp0"

set "CONDA_ENV=subgen"
set "SUBGEN_URL=http://localhost:8501"
set "HF_HOME=%CD%\.cache\huggingface"

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
if defined CONDA_CMD (
    call :resolve_conda_activate
    if defined CONDA_ACTIVATE (
        call "%CONDA_ACTIVATE%" "%CONDA_ENV%" >nul 2>nul
        if not errorlevel 1 (
            set "PYTHON_EXE=python"
            exit /b 0
        )
    )
)
if exist ".venv\Scripts\python.exe" (
    set "PYTHON_EXE=%CD%\.venv\Scripts\python.exe"
)
exit /b 0

:resolve_conda_activate
set "CONDA_ACTIVATE="
for /f "delims=" %%b in ('call "%CONDA_CMD%" info --base 2^>nul') do (
    if not defined CONDA_ACTIVATE set "CONDA_ACTIVATE=%%b\Scripts\activate.bat"
)
if defined CONDA_ACTIVATE if exist "%CONDA_ACTIVATE%" exit /b 0
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
