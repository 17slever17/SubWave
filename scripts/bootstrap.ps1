$ErrorActionPreference = "Stop"

$AppDir = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$VenvDir = Join-Path $AppDir ".venv"
$VenvPython = Join-Path $VenvDir "Scripts\python.exe"
$Requirements = Join-Path $AppDir "requirements.txt"
$StateFile = Join-Path $VenvDir ".install-state"
$DeepFilterLibWheel = Join-Path $AppDir "wheels\DeepFilterLib-0.5.6-cp312-none-win_amd64.whl"

function Find-CompatiblePython {
    $candidates = @(
        @("py", "-3.12"),
        @("python", "")
    )
    foreach ($candidate in $candidates) {
        try {
            $command = Get-Command $candidate[0] -ErrorAction Stop
            $arguments = @()
            if ($candidate[1]) { $arguments += $candidate[1] }
            $arguments += @("-c", "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 12) else 1)")
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
        throw "Python 3.12 is required. Install it from https://www.python.org/downloads/ and run start.bat again."
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
    # Keep winget attached to the console. Redirecting it through PowerShell 5.1
    # makes winget emit UTF-8 that the OEM code page can decode incorrectly.
    $wingetProcess = Start-Process -FilePath $winget.Source -ArgumentList $wingetArgs -NoNewWindow -Wait -PassThru
    $wingetExitCode = $wingetProcess.ExitCode
    if ($wingetExitCode -ne 0) {
        throw "Python installation failed (winget exit code $wingetExitCode). Install Python 3.12 manually and run start.bat again."
    }

    $installed = Join-Path $env:LocalAppData "Programs\Python\Python312\python.exe"
    if (-not (Test-Path -LiteralPath $installed)) {
        throw "Python was installed but could not be located at: $installed"
    }
    return @($installed, "")
}

function Convert-CudaVersionToWheelTag {
    param([AllowEmptyString()][string]$NvidiaSmiOutput)

    $versionMatch = [regex]::Match($NvidiaSmiOutput, "CUDA\s+Version\s*:\s*(\d+)\.(\d+)", "IgnoreCase")
    if (-not $versionMatch.Success) {
        $versionMatch = [regex]::Match($NvidiaSmiOutput, "CUDA\s+UMD\s+Version\s*:\s*(\d+)\.(\d+)", "IgnoreCase")
    }
    if (-not $versionMatch.Success) {
        return $null
    }

    $major = [int]$versionMatch.Groups[1].Value
    $minor = [int]$versionMatch.Groups[2].Value
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

function Get-CudaWheelTag {
    if (-not (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) {
        return $null
    }
    try {
        $output = (& nvidia-smi 2>$null | Out-String)
        return Convert-CudaVersionToWheelTag -NvidiaSmiOutput $output
    }
    catch {
        return $null
    }
}

function Test-VulkanRuntimeAvailable {
    param([string]$WindowsDirectory = $env:WINDIR)

    if ([string]::IsNullOrWhiteSpace($WindowsDirectory)) {
        $WindowsDirectory = [Environment]::GetFolderPath("Windows")
    }
    # Registry and SDK-tool probes can miss valid driver installs; the loader is sufficient to select Vulkan.
    $loaderPath = Join-Path $WindowsDirectory "System32\vulkan-1.dll"
    return Test-Path -LiteralPath $loaderPath -PathType Leaf
}

function Select-LlamaBackend {
    param(
        [bool]$CudaRuntimeAvailable,
        [bool]$VulkanRuntimeAvailable
    )

    if ($CudaRuntimeAvailable) { return "cuda" }
    if ($VulkanRuntimeAvailable) { return "vulkan" }
    return "cpu"
}

function Test-Cuda124RuntimeAvailable {
    param([string]$CudaWheelTag)
    return $CudaWheelTag -in @("cu124", "cu125", "cu130", "cu132")
}

function Get-LlamaBackendFromDirectory {
    param([Parameter(Mandatory = $true)][string]$BinDir)

    if (-not (Test-Path -LiteralPath $BinDir -PathType Container)) {
        return $null
    }

    $dllNames = @(
        Get-ChildItem -LiteralPath $BinDir -Recurse -File -ErrorAction SilentlyContinue |
            ForEach-Object { $_.Name }
    )
    $hasCuda = [bool]($dllNames | Where-Object { $_ -match "^(ggml-cuda|cublas|cudart).*\.dll$" })
    $hasVulkan = [bool]($dllNames | Where-Object { $_ -match "^ggml-vulkan.*\.dll$" })
    if ($hasCuda -and $hasVulkan) { return "mixed" }

    $markerPath = Join-Path $BinDir "runtime-backend.txt"
    $marker = $null
    if (Test-Path -LiteralPath $markerPath -PathType Leaf) {
        $candidateMarker = (Get-Content -LiteralPath $markerPath -Raw).Trim().ToLowerInvariant()
        if ($candidateMarker -in @("cuda", "vulkan", "cpu")) {
            $marker = $candidateMarker
        }
    }
    if ($marker) {
        if (($hasCuda -and $marker -ne "cuda") -or ($hasVulkan -and $marker -ne "vulkan")) {
            return "mixed"
        }
        return $marker
    }
    if ($hasCuda) { return "cuda" }
    if ($hasVulkan) { return "vulkan" }
    if (Test-Path -LiteralPath (Join-Path $BinDir "llama-server.exe") -PathType Leaf) {
        return "cpu"
    }
    return $null
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

function Test-VendoredDeepFilterLibPlatform {
    param([Parameter(Mandatory = $true)][string]$Python)

    & $Python -c "import platform, struct, sys; raise SystemExit(0 if sys.platform == 'win32' and sys.version_info[:2] == (3, 12) and platform.machine().lower() in ('amd64', 'x86_64') and struct.calcsize('P') == 8 else 1)" *> $null
    return $LASTEXITCODE -eq 0
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

function Install-LlamaServer {
    param([ValidateSet("cuda", "vulkan", "cpu")][string]$Backend)

    $binDir = Join-Path $AppDir "bin\llama.cpp"
    $serverExe = Join-Path $binDir "llama-server.exe"
    if (Test-Path -LiteralPath $serverExe -PathType Leaf) {
        $installedBackend = Get-LlamaBackendFromDirectory -BinDir $binDir
        if ($installedBackend -eq $Backend) {
            $markerPath = Join-Path $binDir "runtime-backend.txt"
            [IO.File]::WriteAllText($markerPath, $Backend, [Text.Encoding]::ASCII)
            Write-Host "[setup] Existing llama.cpp runtime backend is $Backend."
            return
        }
        Write-Host "[setup] Replacing llama.cpp runtime ($installedBackend -> $Backend) to avoid mixed backend DLLs."
    }

    Write-Host "[setup] Installing native llama.cpp runtime ($Backend)..."
    $releaseResponse = Invoke-RestMethod `
        -Uri "https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=10" `
        -Headers @{ "User-Agent" = "realtime-translator-installer" }
    $releases = @($releaseResponse.GetEnumerator())
    $suffix = switch ($Backend) {
        "cuda" { "bin-win-cuda-12.4-x64.zip" }
        "vulkan" { "bin-win-vulkan-x64.zip" }
        "cpu" { "bin-win-cpu-x64.zip" }
    }
    $release = $null
    $assets = @()
    foreach ($candidate in $releases) {
        $runtimeAsset = $candidate.assets |
            Where-Object { $_.name -like "llama-*-$suffix" } |
            Select-Object -First 1
        if ($runtimeAsset) {
            $candidateAssets = @($runtimeAsset)
            if ($Backend -eq "cuda") {
                $cudaRuntime = $candidate.assets |
                    Where-Object { $_.name -eq "cudart-llama-bin-win-cuda-12.4-x64.zip" } |
                    Select-Object -First 1
                if (-not $cudaRuntime) {
                    continue
                }
                $candidateAssets += $cudaRuntime
            }
            $release = $candidate
            $assets = $candidateAssets
            break
        }
    }
    if (-not $release -or -not $assets) {
        throw "Recent llama.cpp releases do not contain the expected Windows runtime."
    }
    Write-Host "[setup] Using llama.cpp release $($release.tag_name)."

    $temporaryDir = Join-Path $env:TEMP "realtime-translator-llama-runtime"
    $stagingDir = Join-Path (Split-Path -Parent $binDir) ("llama.cpp.staging-" + [guid]::NewGuid().ToString("N"))
    $backupDir = Join-Path (Split-Path -Parent $binDir) ("llama.cpp.backup-" + [guid]::NewGuid().ToString("N"))
    New-Item -ItemType Directory -Force -Path $temporaryDir | Out-Null
    New-Item -ItemType Directory -Force -Path $stagingDir | Out-Null
    $installSucceeded = $false
    $previousRuntimeMoved = $false
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
                Copy-Item -Destination $stagingDir -Force
        }

        $stagedServerExe = Join-Path $stagingDir "llama-server.exe"
        if (-not (Test-Path -LiteralPath $stagedServerExe -PathType Leaf)) {
            throw "llama-server.exe was not found after extracting the official runtime."
        }
        $stagedBackend = Get-LlamaBackendFromDirectory -BinDir $stagingDir
        if ($stagedBackend -ne $Backend) {
            throw "Downloaded llama.cpp files identify as backend '$stagedBackend', expected '$Backend'."
        }
        [IO.File]::WriteAllText(
            (Join-Path $stagingDir "runtime-backend.txt"),
            $Backend,
            [Text.Encoding]::ASCII
        )

        if (Test-Path -LiteralPath $binDir -PathType Container) {
            Move-Item -LiteralPath $binDir -Destination $backupDir
            $previousRuntimeMoved = $true
        }
        try {
            Move-Item -LiteralPath $stagingDir -Destination $binDir
        }
        catch {
            if ($previousRuntimeMoved -and -not (Test-Path -LiteralPath $binDir)) {
                Move-Item -LiteralPath $backupDir -Destination $binDir
                $previousRuntimeMoved = $false
            }
            throw
        }
        $installSucceeded = $true
        if ($previousRuntimeMoved) {
            Remove-Item -LiteralPath $backupDir -Recurse -Force -ErrorAction SilentlyContinue
        }
    }
    finally {
        if (Test-Path -LiteralPath $stagingDir) {
            Remove-Item -LiteralPath $stagingDir -Recurse -Force -ErrorAction SilentlyContinue
        }
        if ($installSucceeded) {
            Remove-Item -LiteralPath $temporaryDir -Recurse -Force -ErrorAction SilentlyContinue
        }
        else {
            Write-Host "[setup] Keeping partial downloads for the next run: $temporaryDir"
        }
    }

    Write-Host "[setup] Native llama.cpp runtime ($Backend) is ready."
}

Set-Location $AppDir
if (Test-Path $VenvPython) {
    & $VenvPython -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 12) else 1)" *> $null
    if ($LASTEXITCODE -ne 0) {
        throw "The existing .venv uses an unsupported Python version. Delete '$VenvDir' and run start.bat again to recreate it with Python 3.12."
    }
}
else {
    $PythonCommand = Find-CompatiblePython
    if (-not $PythonCommand) {
        $PythonCommand = Install-Python
    }
    Write-Host "[setup] Creating an isolated Python environment..."
    $venvArgs = @()
    if ($PythonCommand[1]) { $venvArgs += $PythonCommand[1] }
    $venvArgs += @("-m", "venv", $VenvDir)
    Invoke-Checked $PythonCommand[0] $venvArgs "Could not create the local Python environment."
}

$requirementsHash = (Get-FileHash -Algorithm SHA256 $Requirements).Hash
$bootstrapHash = (Get-FileHash -Algorithm SHA256 $MyInvocation.MyCommand.Path).Hash
$cudaTag = Get-CudaWheelTag
$cudaRuntimeAvailable = Test-Cuda124RuntimeAvailable -CudaWheelTag $cudaTag
$vulkanRuntimeAvailable = Test-VulkanRuntimeAvailable
$llamaBackend = Select-LlamaBackend `
    -CudaRuntimeAvailable $cudaRuntimeAvailable `
    -VulkanRuntimeAvailable $vulkanRuntimeAvailable
$runtimeTag = if ($cudaTag) { $cudaTag } else { "cpu" }
$useVendoredDeepFilterLib = Test-VendoredDeepFilterLibPlatform -Python $VenvPython
$deepFilterLibWheelHash = "pypi"
if ($useVendoredDeepFilterLib) {
    if (-not (Test-Path -LiteralPath $DeepFilterLibWheel)) {
        throw "The vendored DeepFilterLib wheel is required for Python 3.12 x64: $DeepFilterLibWheel"
    }
    $deepFilterLibWheelHash = (Get-FileHash -Algorithm SHA256 $DeepFilterLibWheel).Hash
}
$desiredState = "$requirementsHash`n$bootstrapHash`n$runtimeTag`n$deepFilterLibWheelHash"
$currentState = if (Test-Path $StateFile) { Get-Content -Raw $StateFile } else { "" }

if ($currentState.Trim() -ne $desiredState.Trim()) {
    Write-Host "[setup] Installing required libraries. This can take several minutes on the first run..."
    Invoke-Checked $VenvPython @("-m", "pip", "install", "--upgrade", "pip", "setuptools") "Could not update pip."
    Invoke-Checked $VenvPython @("-m", "pip", "install", "torch==2.10.0", "torchaudio==2.10.0", "--index-url", "https://download.pytorch.org/whl/cpu") "Could not install PyTorch."
    if ($useVendoredDeepFilterLib) {
        Invoke-Checked $VenvPython @("-m", "pip", "install", "--no-deps", $DeepFilterLibWheel) "Could not install the vendored DeepFilterLib wheel."
    }
    Invoke-Checked $VenvPython @("-m", "pip", "install", "--requirement", $Requirements) "Could not install runtime libraries."

    Set-Content -Encoding ASCII -Path $StateFile -Value $desiredState
    Write-Host "[setup] Installation complete."
}
else {
    Write-Host "[setup] Environment is ready."
}

Install-LlamaServer -Backend $llamaBackend
