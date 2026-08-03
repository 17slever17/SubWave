Add-Type -AssemblyName System.Drawing

$states = @{
    gray = [System.Drawing.Color]::FromArgb(255, 143, 143, 143)
    orange = [System.Drawing.Color]::FromArgb(255, 240, 100, 67)
    blue = [System.Drawing.Color]::FromArgb(255, 62, 166, 255)
    green = [System.Drawing.Color]::FromArgb(255, 70, 200, 120)
}

$outputDirectory = Join-Path $PSScriptRoot "icons"
[System.IO.Directory]::CreateDirectory($outputDirectory) | Out-Null

foreach ($state in $states.GetEnumerator()) {
    foreach ($size in @(16, 32)) {
        $bitmap = New-Object System.Drawing.Bitmap $size, $size
        $graphics = [System.Drawing.Graphics]::FromImage($bitmap)
        $graphics.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::AntiAlias
        $graphics.Clear([System.Drawing.Color]::Transparent)

        $scale = $size / 24.0
        if ($state.Key -ne "gray") {
            $glowColor = [System.Drawing.Color]::FromArgb(54, $state.Value)
            $glowPen = New-Object System.Drawing.Pen $glowColor, ([Math]::Max(2.5, 4.4 * $scale))
            $graphics.DrawEllipse($glowPen, 2.0 * $scale, 2.0 * $scale, 20.0 * $scale, 20.0 * $scale)
            $glowPen.Dispose()
        }

        $pen = New-Object System.Drawing.Pen $state.Value, ([Math]::Max(1.4, 2.0 * $scale))
        $pen.StartCap = [System.Drawing.Drawing2D.LineCap]::Round
        $pen.EndCap = [System.Drawing.Drawing2D.LineCap]::Round
        $graphics.DrawEllipse($pen, 2.0 * $scale, 2.0 * $scale, 20.0 * $scale, 20.0 * $scale)
        $graphics.DrawLine($pen, 12.0 * $scale, 7.0 * $scale, 12.0 * $scale, 11.0 * $scale)
        $graphics.DrawArc($pen, 7.0 * $scale, 7.0 * $scale, 10.0 * $scale, 10.0 * $scale, -143, 286)

        $path = Join-Path $outputDirectory ("power-{0}-{1}.png" -f $state.Key, $size)
        $bitmap.Save($path, [System.Drawing.Imaging.ImageFormat]::Png)
        $pen.Dispose()
        $graphics.Dispose()
        $bitmap.Dispose()
    }
}
