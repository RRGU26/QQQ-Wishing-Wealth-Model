@echo off
REM Batch script to set up Windows Task Scheduler for daily reports
REM Run this as Administrator

echo ============================================================
echo Wishing Wealth Daily Report - Task Scheduler Setup
echo ============================================================
echo.

REM Get the directory of this batch file
set SCRIPT_DIR=%~dp0
set PYTHON_SCRIPT=%SCRIPT_DIR%run_daily.py

REM Task details
set TASK_NAME=Wishing_Wealth_Daily
set TASK_TIME=16:10

echo Creating scheduled task: %TASK_NAME%
echo Script: %PYTHON_SCRIPT%
echo Time: %TASK_TIME% (4:10 PM) on weekdays
echo.

REM Delete existing task if present
schtasks /delete /tn "%TASK_NAME%" /f >nul 2>&1

REM Create the task to run Monday through Friday at 4:05 PM with visible window
set PYTHON_EXE=C:\Users\rrose\AppData\Local\Programs\Python\Python312\python.exe
schtasks /create /tn "%TASK_NAME%" /tr "cmd /k cd /d %SCRIPT_DIR% && \"%PYTHON_EXE%\" \"%PYTHON_SCRIPT%\"" /sc weekly /d MON,TUE,WED,THU,FRI /st %TASK_TIME% /f

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ============================================================
    echo Task created successfully!
    echo ============================================================
    echo.
    echo The report will run daily at 4:10 PM on weekdays.
    echo.
    echo To test immediately:
    echo   schtasks /run /tn "%TASK_NAME%"
    echo.
    echo To view task status:
    echo   schtasks /query /tn "%TASK_NAME%" /v
    echo.
    echo To delete the task:
    echo   schtasks /delete /tn "%TASK_NAME%" /f
) else (
    echo.
    echo ERROR: Failed to create scheduled task.
    echo Make sure you're running as Administrator.
)

echo.
pause
