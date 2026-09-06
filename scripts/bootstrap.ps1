$ErrorActionPreference = "Stop"

$AppDir = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$VenvDir = Join-Path $AppDir ".venv"
$VenvPython = Join-Path $VenvDir "Scripts\python.exe"
$Requirements = Join-Path $AppDir "requirements.txt"
$StateFile = Join-Path $VenvDir ".install-state"

function Find-CompatiblePython {
    $candidates = @(
        @("py", "-3.12"),
        @("py", "-3.11"),
        @("python", "")
    )
    foreach ($candidate in $candidates) {
        try {
            $command = Get-Command $candidate[0] -ErrorAction Stop
            $arguments = @()
            if ($candidate[1]) { $arguments += $candidate[1] }
            $arguments += @("-c", "import sys; raise SystemExit(0 if (3, 11) <= sys.version_info[:2] <= (3, 12) else 1)")
            & $command.Source @arguments *> $null
            if ($LASTEXITCODE -eq 0) {
                return @($command.Source, $candidate[1])
            }
        }
        catch {
            continue
        }
    }
    return $null
}

function Install-Python {
    $winget = Get-Command winget.exe -ErrorAction SilentlyContinue
    if (-not $winget) {
        throw "Python 3.11 or 3.12 is required. Install it from https://www.python.org/downloads/ and run start.bat again."
    }

    Write-Host "[setup] Python was not found. Installing Python 3.12 for the current user..."
    $wingetArgs = @(
        "install",
        "--id", "Python.Python.3.12",
        "--exact",
        "--scope", "user",
        "--accept-package-agreements",
        "--accept-source-agreements",
        "--disable-interactivity"
    )
    & $winget.Source @wingetArgs 2>&1 | Out-Host
    $wingetExitCode = $LASTEXITCODE
    if ($wingetExitCode -ne 0) {
        throw "Python installation failed (winget exit code $wingetExitCode). Install Python 3.12 manually and run start.bat again."
    }

    $installed = Join-Path $env:LocalAppData "Programs\Python\Python312\python.exe"
    if (-not (Test-Path -LiteralPath $installed)) {
        throw "Python was installed but could not be located at: $installed"
    }
    return @($installed, "")
}

function Get-CudaWheelTag {
    if (-not (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) {
        return $null
    }
    try {
        $output = (& nvidia-smi 2>$null | Out-String)
        if ($output -notmatch "CUDA Version:\s*(\d+)\.(\d+)") {
            return "cu124"
        }
        $major = [int]$Matches[1]
        $minor = [int]$Matches[2]
        if ($major -ge 13) {
            if ($major -gt 13 -or $minor -ge 2) { return "cu132" }
            return "cu130"
        }
        if ($major -eq 12) {
            if ($minor -ge 5) { return "cu125" }
            if ($minor -ge 4) { return "cu124" }
            if ($minor -ge 3) { return "cu123" }
            if ($minor -ge 2) { return "cu122" }
            return "cu121"
        }
        return "cu118"
    }
    catch {
        return "cu124"
    }
}

function Invoke-Checked {
    param(
        [Parameter(Mandatory = $true)][string]$Executable,
        [Parameter(Mandatory = $true)][string[]]$Arguments,
        [Parameter(Mandatory = $true)][string]$FailureMessage
    )
    & $Executable @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw $FailureMessage
    }
}

function Invoke-Download {
    param(
        [Parameter(Mandatory = $true)][string]$Uri,
        [Parameter(Mandatory = $true)][string]$Destination,
        [long]$ExpectedBytes = 0
    )

    if (Test-Path -LiteralPath $Destination) {
        $existingBytes = (Get-Item -LiteralPath $Destination).Length
        if ($ExpectedBytes -gt 0 -and $existingBytes -eq $ExpectedBytes) {
            Write-Host "[setup] Using completed download: $([IO.Path]::GetFileName($Destination))"
            return
        }
        if ($ExpectedBytes -gt 0 -and $existingBytes -gt $ExpectedBytes) {
            Remove-Item -LiteralPath $Destination -Force
        }
    }

    $parallelDownloader = Join-Path $AppDir "scripts\download_parallel.py"
    if (
        $ExpectedBytes -gt 0 -and
        (Test-Path -LiteralPath $VenvPython) -and
        (Test-Path -LiteralPath $parallelDownloader)
    ) {
        $connections = 8
        if ($env:REALTIME_TRANSLATOR_DOWNLOAD_CONNECTIONS) {
            $connections = [Math]::Min(
                16,
                [Math]::Max(2, [int]$env:REALTIME_TRANSLATOR_DOWNLOAD_CONNECTIONS)
            )
        }
        Invoke-Checked `
            $VenvPython `
            @(
                $parallelDownloader,
                "--url", $Uri,
                "--output", $Destination,
                "--size", [string]$ExpectedBytes,
                "--connections", [string]$connections
            ) `
            "Parallel download failed: $Uri"
        return
    }

    $curl = Get-Command curl.exe -ErrorAction SilentlyContinue
    if ($curl) {
        & $curl.Source `
            --location `
            --fail `
            --retry 5 `
            --retry-delay 2 `
            --connect-timeout 20 `
            --continue-at - `
            --progress-bar `
            --user-agent "realtime-translator-installer" `
            --output $Destination `
            $Uri
        if ($LASTEXITCODE -ne 0) {
            throw "Download failed: $Uri"
        }
    }
    else {
        Invoke-WebRequest `
            -Uri $Uri `
            -OutFile $Destination `
            -UseBasicParsing `
            -Headers @{ "User-Agent" = "realtime-translator-installer" }
    }

    if ($ExpectedBytes -gt 0) {
        $downloadedBytes = (Get-Item -LiteralPath $Destination).Length
        if ($downloadedBytes -ne $ExpectedBytes) {
            throw "Downloaded file size mismatch: expected $ExpectedBytes bytes, got $downloadedBytes."
        }
    }
}

function Install-LlamaPythonCuda {
    param([Parameter(Mandatory = $true)][string]$CudaTag)

    $releaseResponse = Invoke-RestMethod `
            -Uri "https://api.github.com/repos/abetlen/llama-cpp-python/releases?per_page=100" `
            -Headers @{ "User-Agent" = "realtime-translator-installer" }
    $releases = @($releaseResponse.GetEnumerator())
    $asset = $null
    foreach ($release in $releases) {
        if ($release.tag_name -notlike "*-$CudaTag") {
            continue
        }
        $asset = $release.assets |
            Where-Object {
                $_.name -match '^llama_cpp_python-.*-py3-none-win_amd64\.whl$'
            } |
            Select-Object -First 1
        if ($asset) {
            break
        }
    }
    if (-not $asset) {
        throw "No compatible Windows llama-cpp-python wheel was found for $CudaTag."
    }

    $cacheDir = Join-Path $env:TEMP "realtime-translator-llama-python"
    $wheelPath = Join-Path $cacheDir $asset.name
    New-Item -ItemType Directory -Force -Path $cacheDir | Out-Null
    Write-Host "[setup] Downloading $($asset.name) with parallel connections..."
    Invoke-Download `
        -Uri $asset.browser_download_url `
        -Destination $wheelPath `
        -ExpectedBytes $asset.size
    Invoke-Checked `
        $VenvPython `
        @("-m", "pip", "install", $wheelPath) `
        "CUDA llama-cpp-python wheel installation failed."
    Remove-Item -LiteralPath $cacheDir -Recurse -Force -ErrorAction SilentlyContinue
}

function Install-LlamaServer {
    param([bool]$UseCuda)

    $binDir = Join-Path $AppDir "bin\llama.cpp"
    $serverExe = Join-Path $binDir "llama-server.exe"
    if (Test-Path $serverExe) {
        return
    }

    Write-Host "[setup] Installing native llama.cpp runtime for MTP..."
    $releaseResponse = Invoke-RestMethod `
        -Uri "https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=10" `
        -Headers @{ "User-Agent" = "realtime-translator-installer" }
    $releases = @($releaseResponse.GetEnumerator())
    $suffix = if ($UseCuda) { "bin-win-cuda-12.4-x64.zip" } else { "bin-win-cpu-x64.zip" }
    $release = $null
    $assets = @()
    foreach ($candidate in $releases) {
        $runtimeAsset = $candidate.assets |
            Where-Object { $_.name -like "llama-*-$suffix" } |
            Select-Object -First 1
        if ($runtimeAsset) {
            $release = $candidate
            $assets = @($runtimeAsset)
            if ($UseCuda) {
                $cudaRuntime = $candidate.assets |
                    Where-Object { $_.name -eq "cudart-llama-bin-win-cuda-12.4-x64.zip" } |
                    Select-Object -First 1
                if ($cudaRuntime) {
                    $assets += $cudaRuntime
                }
            }
            break
        }
    }
    if (-not $release -or -not $assets) {
        throw "Recent llama.cpp releases do not contain the expected Windows runtime."
    }
    Write-Host "[setup] Using llama.cpp release $($release.tag_name)."

    $temporaryDir = Join-Path $env:TEMP "realtime-translator-llama-runtime"
    New-Item -ItemType Directory -Force -Path $temporaryDir | Out-Null
    New-Item -ItemType Directory -Force -Path $binDir | Out-Null
    $installSucceeded = $false
    try {
        foreach ($asset in $assets) {
            $archivePath = Join-Path $temporaryDir $asset.name
            Write-Host "[setup] Downloading $($asset.name)..."
            Invoke-Download `
                -Uri $asset.browser_download_url `
                -Destination $archivePath `
                -ExpectedBytes $asset.size
            $extractDir = Join-Path $temporaryDir ([IO.Path]::GetFileNameWithoutExtension($asset.name))
            if (Test-Path -LiteralPath $extractDir) {
                Remove-Item -LiteralPath $extractDir -Recurse -Force
            }
            Expand-Archive -LiteralPath $archivePath -DestinationPath $extractDir -Force
            Get-ChildItem -LiteralPath $extractDir -Recurse -File |
                Copy-Item -Destination $binDir -Force
        }
        $installSucceeded = $true
    }
    finally {
        if ($installSucceeded) {
            Remove-Item -LiteralPath $temporaryDir -Recurse -Force -ErrorAction SilentlyContinue
        }
        else {
            Write-Host "[setup] Keeping partial downloads for the next run: $temporaryDir"
        }
    }

    if (-not (Test-Path $serverExe)) {
        throw "llama-server.exe was not found after extracting the official runtime."
    }
    Write-Host "[setup] Native llama.cpp runtime is ready."
}

Set-Location $AppDir
$PythonCommand = Find-CompatiblePython
if (-not $PythonCommand) {
    $PythonCommand = Install-Python
}

if (-not (Test-Path $VenvPython)) {
    Write-Host "[setup] Creating an isolated Python environment..."
    $venvArgs = @()
    if ($PythonCommand[1]) { $venvArgs += $PythonCommand[1] }
    $venvArgs += @("-m", "venv", $VenvDir)
    Invoke-Checked $PythonCommand[0] $venvArgs "Could not create the local Python environment."
}

$requirementsHash = (Get-FileHash -Algorithm SHA256 $Requirements).Hash
$bootstrapHash = (Get-FileHash -Algorithm SHA256 $MyInvocation.MyCommand.Path).Hash
$cudaTag = Get-CudaWheelTag
$runtimeTag = if ($cudaTag) { $cudaTag } else { "cpu" }
$desiredState = "$requirementsHash`n$bootstrapHash`n$runtimeTag"
$currentState = if (Test-Path $StateFile) { Get-Content -Raw $StateFile } else { "" }

if ($currentState.Trim() -ne $desiredState.Trim()) {
    Write-Host "[setup] Installing required libraries. This can take several minutes on the first run..."
    Invoke-Checked $VenvPython @("-m", "pip", "install", "--upgrade", "pip", "setuptools") "Could not update pip."
    Invoke-Checked $VenvPython @("-m", "pip", "install", "torch==2.10.0", "torchaudio==2.10.0", "--index-url", "https://download.pytorch.org/whl/cpu") "Could not install PyTorch."
    Invoke-Checked $VenvPython @("-m", "pip", "install", "--requirement", $Requirements) "Could not install runtime libraries."

    if ($cudaTag) {
        Write-Host "[setup] NVIDIA GPU detected. Installing llama.cpp CUDA wheel ($cudaTag)..."
        try {
            Install-LlamaPythonCuda -CudaTag $cudaTag
        }
        catch {
            Write-Warning "Parallel CUDA wheel installation failed: $($_.Exception.Message)"
            Write-Host "[setup] Retrying with pip's standard downloader..."
            & $VenvPython -m pip install llama-cpp-python --only-binary=llama-cpp-python --extra-index-url "https://abetlen.github.io/llama-cpp-python/whl/$cudaTag"
        }
    }
    if (-not $cudaTag -or $LASTEXITCODE -ne 0) {
        Write-Host "[setup] Installing portable llama.cpp CPU wheel..."
        & $VenvPython -m pip install llama-cpp-python --only-binary=llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
    }
    if ($LASTEXITCODE -ne 0) {
        throw "llama-cpp-python installation failed. See the messages above."
    }

    Set-Content -Encoding ASCII -Path $StateFile -Value $desiredState
    Write-Host "[setup] Installation complete."
}
else {
    Write-Host "[setup] Environment is ready."
}

Install-LlamaServer -UseCuda ([bool]$cudaTag)
