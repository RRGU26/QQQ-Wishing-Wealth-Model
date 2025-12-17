# PowerShell script to set up Windows Task Scheduler for daily predictions
# Run this script as Administrator

$TaskName = "QQQ_Wishing_Wealth_Daily"
$Description = "Run QQQ Wishing Wealth prediction model daily after market close"

# Path to Python and script
$PythonPath = "python"  # Or full path like "C:\Python311\python.exe"
$ScriptPath = Join-Path $PSScriptRoot "src\daily_runner.py"
$WorkingDir = $PSScriptRoot

# Create the action
$Action = New-ScheduledTaskAction -Execute $PythonPath -Argument "`"$ScriptPath`"" -WorkingDirectory $WorkingDir

# Create trigger - Run at 4:30 PM ET (after market close) on weekdays
# Adjust time based on your timezone
$Trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday,Tuesday,Wednesday,Thursday,Friday -At "4:30PM"

# Create settings
$Settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -DontStopOnIdleEnd -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries

# Register the task
try {
    # Check if task already exists
    $ExistingTask = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue

    if ($ExistingTask) {
        Write-Host "Task '$TaskName' already exists. Updating..." -ForegroundColor Yellow
        Set-ScheduledTask -TaskName $TaskName -Action $Action -Trigger $Trigger -Settings $Settings
    } else {
        Write-Host "Creating new task '$TaskName'..." -ForegroundColor Green
        Register-ScheduledTask -TaskName $TaskName -Action $Action -Trigger $Trigger -Settings $Settings -Description $Description
    }

    Write-Host ""
    Write-Host "Task scheduled successfully!" -ForegroundColor Green
    Write-Host "The model will run daily at 4:30 PM on weekdays."
    Write-Host ""
    Write-Host "To test immediately, run:" -ForegroundColor Cyan
    Write-Host "  Start-ScheduledTask -TaskName '$TaskName'"
    Write-Host ""
    Write-Host "To view task status:" -ForegroundColor Cyan
    Write-Host "  Get-ScheduledTask -TaskName '$TaskName' | Get-ScheduledTaskInfo"
    Write-Host ""
    Write-Host "To remove the task:" -ForegroundColor Cyan
    Write-Host "  Unregister-ScheduledTask -TaskName '$TaskName' -Confirm:`$false"

} catch {
    Write-Host "Error creating scheduled task: $_" -ForegroundColor Red
    Write-Host ""
    Write-Host "Make sure you're running PowerShell as Administrator." -ForegroundColor Yellow
}
