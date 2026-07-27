@echo off
chcp 65001 >nul
setlocal EnableExtensions
cd /d "%~dp0"

set "CONDA_ENV=subgen"
set "HF_HOME=%CD%\.cache\huggingface"
set "PIP_DISABLE_PIP_VERSION_CHECK=1"

echo ============================================
echo   Subtitle Generator - first-time setup
echo ============================================
echo Project: %CD%
echo Model cache: %HF_HOME%
echo.

call :find_conda
if defined CONDA_CMD (
    echo Found Conda: %CONDA_CMD%
    call :resolve_conda_activate
    if not defined CONDA_ACTIVATE (
        echo Could not locate Conda activate.bat. Falling back to a project-local .venv.
        call :setup_venv
        goto :setup_ready
    )
    call :setup_conda
) else (
    echo Conda was not found. Falling back to a project-local .venv.
    call :setup_venv
)
:setup_ready
if errorlevel 1 goto :failed

echo.
echo Installing Subtitle Generator dependencies...
call "%PYTHON_EXE%" -m pip install --upgrade pip
if errorlevel 1 goto :failed
call "%PYTHON_EXE%" -m pip install --prefer-binary -e ".[ui,translate,test]"
if errorlevel 1 goto :failed

echo.
echo Preparing translation model...
call "%PYTHON_EXE%" scripts\download_models.py
if errorlevel 1 goto :failed

echo.
echo Preparing default Whisper model: small
call "%PYTHON_EXE%" -c "from subgen.transcribe import download_model; download_model(model_name='small', device='cpu', compute_type='default')"
if errorlevel 1 goto :failed

echo.
echo ============================================
echo Setup complete.
echo Double-click launch.bat or 启动字幕生成器.bat to start.
echo You can download or use medium later from the UI or CLI.
echo ============================================
pause
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

:setup_conda
call "%CONDA_CMD%" env list | findstr /r /c:"^%CONDA_ENV%[ ]" >nul 2>nul
if errorlevel 1 (
    echo Creating Conda environment: %CONDA_ENV%
    call "%CONDA_CMD%" env create -f environment.yml
) else (
    echo Updating Conda environment: %CONDA_ENV%
    call "%CONDA_CMD%" env update -n "%CONDA_ENV%" -f environment.yml --prune
)
if errorlevel 1 exit /b 1
call "%CONDA_ACTIVATE%" "%CONDA_ENV%"
if errorlevel 1 exit /b 1
set "PYTHON_EXE=python"
exit /b 0

:setup_venv
set "BASE_PYTHON="
py -3.12 -c "import sys; raise SystemExit(sys.version_info < (3, 12))" >nul 2>nul
if not errorlevel 1 set "BASE_PYTHON=py -3.12"
if not defined BASE_PYTHON (
    python -c "import sys; raise SystemExit(sys.version_info < (3, 12))" >nul 2>nul
    if not errorlevel 1 set "BASE_PYTHON=python"
)
if not defined BASE_PYTHON (
    echo Python 3.12+ was not found.
    echo Install Miniconda or Python 3.12+, then run setup.bat again.
    exit /b 1
)
if not exist ".venv\Scripts\python.exe" (
    echo Creating local virtual environment: .venv
    %BASE_PYTHON% -m venv .venv
    if errorlevel 1 exit /b 1
)
set "PYTHON_EXE=%CD%\.venv\Scripts\python.exe"
exit /b 0

:failed
echo.
echo Setup failed. Check the message above, then rerun setup.bat.
pause
exit /b 1
