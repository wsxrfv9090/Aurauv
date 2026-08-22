@echo off
setlocal EnableExtensions

where uv >nul 2>nul
if errorlevel 1 (
  echo [aurauv] ERROR: uv was not found on PATH. 1>&2
  echo [aurauv] No project environment or metadata was changed. 1>&2
  exit /b 1
)

set "ENTRY=%~dp0aurauv.py"
if defined AURAUV_BOOTSTRAP_PYTHON (
  "%AURAUV_BOOTSTRAP_PYTHON%" -c "import sys; raise SystemExit(sys.version_info ^< (3, 11))" >nul 2>nul
  if not errorlevel 1 (
    "%AURAUV_BOOTSTRAP_PYTHON%" "%ENTRY%" %*
    exit /b %ERRORLEVEL%
  )
)
where py >nul 2>nul
if not errorlevel 1 (
  py -3 -c "import sys; raise SystemExit(sys.version_info ^< (3, 11))" >nul 2>nul
  if not errorlevel 1 (
    py -3 "%ENTRY%" %*
    exit /b %ERRORLEVEL%
  )
)
where python >nul 2>nul
if not errorlevel 1 (
  python -c "import sys; raise SystemExit(sys.version_info ^< (3, 11))" >nul 2>nul
  if not errorlevel 1 (
    python "%ENTRY%" %*
    exit /b %ERRORLEVEL%
  )
)
echo [aurauv] ERROR: Aurauv requires an existing Python ^>= 3.11 to start. 1>&2
exit /b 1
