[CmdletBinding()]
param([switch]$RequireVulkan)

$ErrorActionPreference = "Stop"

$bootstrapPath = Join-Path (Split-Path -Parent $PSScriptRoot) "scripts\bootstrap.ps1"
$tokens = $null
$parseErrors = $null
$bootstrapAst = [System.Management.Automation.Language.Parser]::ParseFile(
    $bootstrapPath,
    [ref]$tokens,
    [ref]$parseErrors
)
if ($parseErrors.Count -gt 0) {
    throw "bootstrap.ps1 has PowerShell parse errors: $($parseErrors -join '; ')"
}

foreach ($functionName in @(
    "Test-VulkanRuntimeAvailable",
    "Get-CudaWheelTag",
    "Select-LlamaBackend",
    "Convert-CudaVersionToWheelTag",
    "Test-Cuda124RuntimeAvailable",
    "Get-LlamaBackendFromDirectory"
)) {
    $functionAst = $bootstrapAst.Find(
        {
            param($node)
            $node -is [System.Management.Automation.Language.FunctionDefinitionAst] -and
                $node.Name -eq $functionName
        },
        $true
    )
    if (-not $functionAst) {
        throw "Could not locate $functionName in bootstrap.ps1."
    }
    Invoke-Expression $functionAst.Extent.Text
}

function Assert-Equal {
    param($Expected, $Actual, [string]$Description)
    if ($Expected -ne $Actual) {
        throw "$Description`: expected '$Expected', got '$Actual'."
    }
}

Assert-Equal "cuda" (Select-LlamaBackend -CudaRuntimeAvailable $true -VulkanRuntimeAvailable $true) "CUDA precedence"
Assert-Equal "vulkan" (Select-LlamaBackend -CudaRuntimeAvailable $false -VulkanRuntimeAvailable $true) "Vulkan fallback"
Assert-Equal "cpu" (Select-LlamaBackend -CudaRuntimeAvailable $false -VulkanRuntimeAvailable $false) "CPU fallback"
Assert-Equal $false (Test-Cuda124RuntimeAvailable "cu123") "CUDA 12.3 rejection"
Assert-Equal $false (Test-Cuda124RuntimeAvailable "cu118") "Legacy CUDA rejection"
Assert-Equal $true (Test-Cuda124RuntimeAvailable "cu124") "CUDA 12.4 acceptance"
Assert-Equal $true (Test-Cuda124RuntimeAvailable "cu132") "CUDA 13.x acceptance"
Assert-Equal "cu123" (Convert-CudaVersionToWheelTag "| CUDA Version : 12.3 |") "CUDA Version parsing"
Assert-Equal "cu132" (Convert-CudaVersionToWheelTag "| CUDA UMD Version : 13.3 |") "CUDA UMD Version parsing"
Assert-Equal $null (Convert-CudaVersionToWheelTag "nvidia-smi reported no runtime version") "Missing CUDA version"

$nvidiaSmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if ($nvidiaSmi) {
    $hostNvidiaSmiOutput = (& $nvidiaSmi.Source 2>$null | Out-String)
    $hostCudaTag = Convert-CudaVersionToWheelTag $hostNvidiaSmiOutput
    if ($hostNvidiaSmiOutput -match "CUDA\s+(UMD\s+)?Version\s*:") {
        if (-not $hostCudaTag) {
            throw "Could not parse this host's nvidia-smi CUDA version output."
        }
        Assert-Equal $hostCudaTag (Get-CudaWheelTag) "Host CUDA tag"
        if (Test-Cuda124RuntimeAvailable $hostCudaTag) {
            Assert-Equal "cuda" (Select-LlamaBackend -CudaRuntimeAvailable $true -VulkanRuntimeAvailable $true) "Host CUDA backend priority"
        }
        Write-Host "Host nvidia-smi CUDA runtime tag detected: $hostCudaTag"
    }
}

$tempParent = [IO.Path]::GetFullPath([IO.Path]::GetTempPath()).TrimEnd([IO.Path]::DirectorySeparatorChar)
$tempRoot = Join-Path $tempParent ("realtime-translator-backend-test-" + [guid]::NewGuid().ToString("N"))
if ([IO.Path]::GetDirectoryName([IO.Path]::GetFullPath($tempRoot)) -ne $tempParent) {
    throw "Refusing to use a test directory outside the system temporary directory."
}

try {
    New-Item -ItemType Directory -Path $tempRoot | Out-Null
    $serverExe = Join-Path $tempRoot "llama-server.exe"
    [IO.File]::WriteAllBytes($serverExe, [byte[]]@())
    $markerPath = Join-Path $tempRoot "runtime-backend.txt"

    [IO.File]::WriteAllText($markerPath, "cpu", [Text.Encoding]::ASCII)
    Assert-Equal "cpu" (Get-LlamaBackendFromDirectory -BinDir $tempRoot) "CPU marker detection"

    [IO.File]::WriteAllText((Join-Path $tempRoot "ggml-vulkan.dll"), "")
    Assert-Equal "mixed" (Get-LlamaBackendFromDirectory -BinDir $tempRoot) "Stale DLL/marker conflict"
    [IO.File]::WriteAllText($markerPath, "vulkan", [Text.Encoding]::ASCII)
    Assert-Equal "vulkan" (Get-LlamaBackendFromDirectory -BinDir $tempRoot) "Vulkan DLL detection"

    [IO.File]::WriteAllText((Join-Path $tempRoot "ggml-cuda.dll"), "")
    Assert-Equal "mixed" (Get-LlamaBackendFromDirectory -BinDir $tempRoot) "Mixed DLL detection"

    Remove-Item -LiteralPath (Join-Path $tempRoot "ggml-cuda.dll") -Force
    Remove-Item -LiteralPath (Join-Path $tempRoot "ggml-vulkan.dll") -Force
    Remove-Item -LiteralPath $markerPath -Force
    Assert-Equal "cpu" (Get-LlamaBackendFromDirectory -BinDir $tempRoot) "Legacy CPU detection"
}
finally {
    if (Test-Path -LiteralPath $tempRoot) {
        Remove-Item -LiteralPath $tempRoot -Recurse -Force
    }
}

$hostHasVulkan = Test-VulkanRuntimeAvailable
if (-not $hostHasVulkan -and $RequireVulkan) {
    throw "The host Vulkan loader/ICD probe did not detect an available hardware Vulkan runtime."
}
if ($hostHasVulkan) {
    Write-Host "Host Vulkan runtime detected."
}
else {
    Write-Host "Host Vulkan runtime not detected; CPU fallback remains valid."
}

Write-Host "Bootstrap backend tests passed."
