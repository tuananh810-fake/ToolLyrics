param(
    [string]$Host = "127.0.0.1",
    [int]$Port = 5000
)

$env:HOST = $Host
$env:PORT = [string]$Port

if (-not (Get-Command npm.cmd -ErrorAction SilentlyContinue)) {
    throw "npm.cmd was not found on PATH."
}

& npm.cmd start
exit $LASTEXITCODE
