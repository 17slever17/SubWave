$ErrorActionPreference = "Stop"

$AppDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $AppDir ".venv\Scripts\python.exe"
$FrontendDir = Join-Path $AppDir "frontend"
$BackendPort = if ($env:REALTIME_WEBUI_BACKEND_PORT) { [int]$env:REALTIME_WEBUI_BACKEND_PORT } else { 7860 }
$FrontendPort = if ($env:REALTIME_WEBUI_DEV_PORT) { [int]$env:REALTIME_WEBUI_DEV_PORT } else { 5174 }

function Test-HasActiveDevLauncher {
    param([int]$ProcessId)

    $current = Get-CimInstance Win32_Process -Filter "ProcessId = $ProcessId" -ErrorAction SilentlyContinue
    while ($current -and $current.ParentProcessId) {
        $parent = Get-CimInstance Win32_Process -Filter "ProcessId = $($current.ParentProcessId)" -ErrorAction SilentlyContinue
        if (-not $parent) {
            return $false
        }
        if ($parent.CommandLine -like "*start-dev.ps1*") {
            return $true
        }
        $current = $parent
    }
    return $false
}

function Stop-StaleRealtimeBackend {
    param([int]$Port)

    $connections = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
    if (-not $connections) {
        return
    }

    try {
        $openApi = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/openapi.json" -TimeoutSec 2
    }
    catch {
        return
    }

    if ($openApi.info.title -ne "Realtime Subtitle Translator WebUI") {
        return
    }

    $processIds = $connections | Select-Object -ExpandProperty OwningProcess -Unique
    foreach ($processId in $processIds) {
        if (Test-HasActiveDevLauncher -ProcessId $processId) {
            return
        }
    }

    foreach ($processId in $processIds) {
        Write-Host "[start-dev] Stopping stale translator backend pid=$processId on port $Port..."
        & taskkill /PID $processId /T /F *> $null
    }
    Start-Sleep -Milliseconds 300
}

function Stop-StaleRealtimeProcesses {
    $escapedAppDir = [Regex]::Escape($AppDir)
    $processes = Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" -ErrorAction SilentlyContinue |
        Where-Object {
            $_.CommandLine -match $escapedAppDir -and
            $_.CommandLine -like "*main.py*"
        }

    foreach ($process in $processes) {
        if (Test-HasActiveDevLauncher -ProcessId $process.ProcessId) {
            continue
        }
        Write-Host "[start-dev] Stopping stale translator runtime pid=$($process.ProcessId)..."
        & taskkill /PID $process.ProcessId /T /F *> $null
    }
}

function Stop-StaleRealtimeFrontend {
    param([int]$Port)

    $escapedAppDir = [Regex]::Escape($AppDir)
    $connections = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
    foreach ($connection in $connections) {
        $process = Get-CimInstance Win32_Process -Filter "ProcessId = $($connection.OwningProcess)" -ErrorAction SilentlyContinue
        if (-not $process -or $process.CommandLine -notmatch $escapedAppDir -or $process.CommandLine -notlike "*vite*") {
            continue
        }
        if (Test-HasActiveDevLauncher -ProcessId $process.ProcessId) {
            continue
        }
        Write-Host "[start-dev] Stopping stale Vite pid=$($process.ProcessId) on port $Port..."
        & taskkill /PID $process.ProcessId /T /F *> $null
    }
}

Set-Location $AppDir

Write-Host "[start-dev] Checking the local environment..."
& (Join-Path $AppDir "scripts\bootstrap.ps1")

Write-Host "[start-dev] Starting backend and Vite dev server..."
Stop-StaleRealtimeProcesses
Stop-StaleRealtimeBackend -Port $BackendPort
Stop-StaleRealtimeFrontend -Port $FrontendPort

$backendListener = Get-NetTCPConnection -LocalPort $BackendPort -State Listen -ErrorAction SilentlyContinue
if ($backendListener) {
    throw "Backend port $BackendPort is already in use. Set REALTIME_WEBUI_BACKEND_PORT to another free port."
}

$frontendListener = Get-NetTCPConnection -LocalPort $FrontendPort -State Listen -ErrorAction SilentlyContinue
if ($frontendListener) {
    throw "Frontend port $FrontendPort is already in use. Set REALTIME_WEBUI_DEV_PORT to another free port."
}

if (-not (Test-Path (Join-Path $FrontendDir "node_modules"))) {
    Push-Location $FrontendDir
    try {
        & npm install
    }
    finally {
        Pop-Location
    }
}

& $Python (Join-Path $AppDir "scripts\launch_dev.py")
exit $LASTEXITCODE
