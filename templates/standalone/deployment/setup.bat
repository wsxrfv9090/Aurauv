@echo off
setlocal EnableExtensions

set "SETUP=%~dp0setup.py"
if defined AURAUV_BOOTSTRAP_PYTHON (
  "%AURAUV_BOOTSTRAP_PYTHON%" -c "import sys; raise SystemExit(sys.version_info < (3, 11))" >nul 2>nul
  if not errorlevel 1 (
    "%AURAUV_BOOTSTRAP_PYTHON%" "%SETUP%" %*
    exit /b %ERRORLEVEL%
  )
)
where py >nul 2>nul
if not errorlevel 1 (
  py -3 -c "import sys; raise SystemExit(sys.version_info < (3, 11))" >nul 2>nul
  if not errorlevel 1 (
    py -3 "%SETUP%" %*
    exit /b %ERRORLEVEL%
  )
)
where python >nul 2>nul
if not errorlevel 1 (
  python -c "import sys; raise SystemExit(sys.version_info < (3, 11))" >nul 2>nul
  if not errorlevel 1 (
    python "%SETUP%" %*
    exit /b %ERRORLEVEL%
  )
)
echo [aurauv-launcher] Python ^>= 3.11 was not found on PATH. 1>&2
echo [aurauv-launcher] Set AURAUV_BOOTSTRAP_PYTHON to a compatible interpreter. 1>&2
exit /b 1
