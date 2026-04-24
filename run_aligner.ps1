param(
    [Parameter(Mandatory = $true)]
    [string]$Audio,

    [Parameter(Mandatory = $true)]
    [string]$Lrc,

    [string]$Output = "lyrics_pro.json",
    [string]$Song,

    [ValidateSet("recommended", "large-v3", "medium")]
    [string]$Model = "recommended",

    [string]$Language = "vi",
    [string]$Device = "auto",
    [ValidateSet("auto", "direct", "transcribe")]
    [string]$Strategy = "auto",
    [string]$DebugDir
)

$bundledPython = "C:\Users\admin\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
$scriptPath = Join-Path $PSScriptRoot "high_precision_word_aligner.py"

if (-not (Test-Path $bundledPython)) {
    throw "Bundled Python not found at $bundledPython"
}

$argList = @(
    $scriptPath,
    "--audio", $Audio,
    "--lrc", $Lrc,
    "--output", $Output,
    "--model", $Model,
    "--device", $Device,
    "--strategy", $Strategy
)

if ($Song) {
    $argList += @("--song", $Song)
}

if ($Language) {
    $argList += @("--language", $Language)
}

if ($DebugDir) {
    $argList += @("--debug-dir", $DebugDir)
}

& $bundledPython @argList
exit $LASTEXITCODE
