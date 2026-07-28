@echo off
chcp 65001 >nul
setlocal EnableExtensions
cd /d "%~dp0"

set "CONDA_ENV=subgen"
set "HF_HOME=%CD%\.cache\huggingface"
set "CONDA_PKGS_DIRS=%CD%\.conda\pkgs"
set "CONDA_ENVS_PATH=%CD%\.conda\envs"
set "CONDA_NOTICES=false"
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
    if errorlevel 1 (
        echo Conda setup failed. Falling back to a project-local .venv.
        call :setup_venv
    )
) else (
    echo Conda was not found. Falling back to a project-local .venv.
    call :setup_venv
)
:setup_ready
if errorlevel 1 goto :failed

echo.
call :install_dependencies
if errorlevel 1 (
    if "%ENV_KIND%"=="conda" (
        echo Conda package installation failed. Falling back to a project-local .venv.
        call :setup_venv
        if errorlevel 1 goto :failed
        call :install_dependencies
    )
)
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
set "ENV_KIND=conda"
if not exist "logs" mkdir "logs"
set "CONDA_ENV_EXISTS="
call "%CONDA_CMD%" env list > "%TEMP%\subgen-conda-envs.txt" 2>nul
findstr /r /c:"^%CONDA_ENV%[ ]" "%TEMP%\subgen-conda-envs.txt" >nul 2>nul
if errorlevel 1 (
    echo Creating Conda environment: %CONDA_ENV%
    call "%CONDA_CMD%" env create -f environment.yml > "logs\setup-conda.log" 2>&1
) else (
    set "CONDA_ENV_EXISTS=1"
    echo Updating Conda environment: %CONDA_ENV%
    call "%CONDA_CMD%" env update -n "%CONDA_ENV%" -f environment.yml --prune > "logs\setup-conda.log" 2>&1
)
if errorlevel 1 (
    if defined CONDA_ENV_EXISTS (
        echo Conda environment update failed. Using the existing %CONDA_ENV% environment.
        echo Details were saved to logs\setup-conda.log.
    ) else (
        exit /b 1
    )
)
call "%CONDA_ACTIVATE%" "%CONDA_ENV%"
if errorlevel 1 exit /b 1
set "PYTHON_EXE="
for /f "delims=" %%p in ('where python 2^>nul') do (
    if not defined PYTHON_EXE set "PYTHON_EXE=%%p"
)
if not defined PYTHON_EXE exit /b 1
exit /b 0

:setup_venv
set "ENV_KIND=venv"
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

:install_dependencies
echo Installing Subtitle Generator dependencies...
echo Python: %PYTHON_EXE%
call "%PYTHON_EXE%" -m pip --version
if errorlevel 1 exit /b 1
call "%PYTHON_EXE%" -m pip install --prefer-binary "setuptools>=69" wheel
if errorlevel 1 exit /b 1
call "%PYTHON_EXE%" -m pip install --no-build-isolation --prefer-binary -e ".[ui,translate,test]"
if errorlevel 1 exit /b 1
exit /b 0

:failed
echo.
echo Setup failed. Check the message above, then rerun setup.bat.
pause
exit /b 1
