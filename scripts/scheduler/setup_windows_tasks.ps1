# VieComRec Recommendation System - Windows Task Scheduler Setup
# Run this script as Administrator to set up scheduled tasks

# Configuration
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$PythonPath = "python"  # Change to full path if needed, e.g., "C:\Python311\python.exe"
$LogDir = Join-Path $ProjectRoot "logs\scheduler"

# Create log directory
if (!(Test-Path $LogDir)) {
    New-Item -ItemType Directory -Path $LogDir -Force | Out-Null
}

# Function to create a scheduled task
function Create-RecSysTask {
    param(
        [string]$TaskName,
        [string]$Script,
        [string]$Arguments = "",
        [string]$Schedule,
        [string]$Time = "02:00",
        [string]$Description
    )
    
    $ScriptPath = Join-Path $ProjectRoot "scripts\$Script"
    $LogFile = Join-Path $LogDir "$($TaskName.Replace(' ', '_')).log"
    
    # Build action
    $Action = New-ScheduledTaskAction `
        -Execute $PythonPath `
        -Argument "$ScriptPath $Arguments >> `"$LogFile`" 2>&1" `
        -WorkingDirectory $ProjectRoot
    
    # Build trigger based on schedule
    switch ($Schedule) {
        "Daily" {
            $Trigger = New-ScheduledTaskTrigger -Daily -At $Time
        }
        "Weekly" {
            $Trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Sunday -At $Time
        }
        "Hourly" {
            $Trigger = New-ScheduledTaskTrigger -Once -At (Get-Date) `
                -RepetitionInterval (New-TimeSpan -Hours 1) `
                -RepetitionDuration (New-TimeSpan -Days 365)
        }
        default {
            $Trigger = New-ScheduledTaskTrigger -Daily -At $Time
        }
    }
    
    # Task settings
    $Settings = New-ScheduledTaskSettingsSet `
        -AllowStartIfOnBatteries `
        -DontStopIfGoingOnBatteries `
        -StartWhenAvailable `
        -RestartCount 3 `
        -RestartInterval (New-TimeSpan -Minutes 5)
    
    # Register task
    $TaskPath = "\VieComRec\"
    
    try {
        # Remove existing task if it exists
        $ExistingTask = Get-ScheduledTask -TaskPath $TaskPath -TaskName $TaskName -ErrorAction SilentlyContinue
        if ($ExistingTask) {
            Unregister-ScheduledTask -TaskPath $TaskPath -TaskName $TaskName -Confirm:$false
            Write-Host "  Updated existing task: $TaskName" -ForegroundColor Yellow
        }
        
        Register-ScheduledTask `
            -TaskPath $TaskPath `
            -TaskName $TaskName `
            -Action $Action `
            -Trigger $Trigger `
            -Settings $Settings `
            -Description $Description `
            -Force | Out-Null
        
        Write-Host "  Created task: $TaskName ($Schedule at $Time)" -ForegroundColor Green
    }
    catch {
        Write-Host "  Failed to create task $TaskName : $_" -ForegroundColor Red
    }
}

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "VieComRec Scheduler Setup" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

Write-Host "Project Root: $ProjectRoot"
Write-Host "Python: $PythonPath"
Write-Host "Log Directory: $LogDir`n"

# Verify Python is accessible
try {
    $PythonVersion = & $PythonPath --version 2>&1
    Write-Host "Python Version: $PythonVersion`n" -ForegroundColor Green
}
catch {
    Write-Host "ERROR: Python not found at $PythonPath" -ForegroundColor Red
    Write-Host "Please update the PythonPath variable in this script" -ForegroundColor Yellow
    exit 1
}

Write-Host "Creating scheduled tasks...`n"

# 1. Daily Data Refresh (2:00 AM)
Create-RecSysTask `
    -TaskName "Data Refresh" `
    -Script "refresh_data.py" `
    -Arguments "" `
    -Schedule "Daily" `
    -Time "02:00" `
    -Description "Refresh processed data from raw sources"

# 2. Weekly Model Training (Sunday 3:00 AM)
Create-RecSysTask `
    -TaskName "Model Training" `
    -Script "train_both_models.py" `
    -Arguments "--auto-select" `
    -Schedule "Weekly" `
    -Time "03:00" `
    -Description "Train ALS and BPR models weekly"

# 3. Daily Model Deployment (4:00 AM)
Create-RecSysTask `
    -TaskName "Model Deployment" `
    -Script "deploy_model_update.py" `
    -Arguments "" `
    -Schedule "Daily" `
    -Time "04:00" `
    -Description "Deploy best model to production service"

# 4. Hourly Health Check
Create-RecSysTask `
    -TaskName "Health Check" `
    -Script "health_check.py" `
    -Arguments "--alert" `
    -Schedule "Hourly" `
    -Time "00:00" `
    -Description "Run health checks on all components"

# 5. Weekly Cleanup (Sunday 5:00 AM)
Create-RecSysTask `
    -TaskName "Log Cleanup" `
    -Script "cleanup_logs.py" `
    -Arguments "--days 30 --keep-models 5" `
    -Schedule "Weekly" `
    -Time "05:00" `
    -Description "Clean up old logs and artifacts"

Write-Host "`n========================================"
Write-Host "Scheduled Tasks Created:" -ForegroundColor Cyan
Write-Host "========================================"

# List all VieComRec tasks
Get-ScheduledTask -TaskPath "\VieComRec\" | ForEach-Object {
    $Trigger = $_.Triggers | Select-Object -First 1
    Write-Host "  - $($_.TaskName): $($_.State)" -ForegroundColor $(if ($_.State -eq "Ready") { "Green" } else { "Yellow" })
}

Write-Host "`n========================================"
Write-Host "Management Commands:" -ForegroundColor Cyan
Write-Host "========================================"
Write-Host "  View tasks:    Get-ScheduledTask -TaskPath '\VieComRec\'"
Write-Host "  Run manually:  Start-ScheduledTask -TaskPath '\VieComRec\' -TaskName 'Health Check'"
Write-Host "  Disable:       Disable-ScheduledTask -TaskPath '\VieComRec\' -TaskName 'TaskName'"
Write-Host "  Remove all:    Get-ScheduledTask -TaskPath '\VieComRec\' | Unregister-ScheduledTask -Confirm:`$false"
Write-Host ""
