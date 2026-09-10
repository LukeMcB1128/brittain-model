# Start the BRITTAIN-4 server so it outlives the shell that launched it.
#
# A `wsl bash -lc ...` started from an interactive session is a child of that
# session, and WSL reaps it when the session ends -- which is how the server has
# been dying. Start-Process detaches it into a Windows-owned process instead, so
# it keeps running after the terminal, the agent session, or the IDE closes.
#
# Also starts the keep-awake holder, since a server that sleeps with the machine
# is no more useful than one that exited. That has to be a Windows process too:
# SetThreadExecutionState is a Win32 call and a WSL process holding it holds
# nothing.
#
#   powershell -ExecutionPolicy Bypass -File scripts\inference\start_brittain4.ps1
#   ... -NoKeepAwake     start only the server
#   ... -Stop            stop both
[CmdletBinding()]
param(
    [switch]$Stop,
    [switch]$NoKeepAwake
)

$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$health = 'http://localhost:11435/health'

function Test-Server {
    try {
        $r = Invoke-WebRequest -Uri $health -TimeoutSec 3 -UseBasicParsing
        return $r.StatusCode -eq 200
    } catch { return $false }
}

if ($Stop) {
    # Split the pattern so pkill cannot match the command line that carries it.
    wsl bash -lc 'pkill -9 -f "bin/vll""m serve" 2>/dev/null; pkill -9 -f "VLLM::Engine""Core" 2>/dev/null; sleep 3; echo stopped'
    Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
        Where-Object { $_.CommandLine -like '*keep_awake_while_serving*' } |
        ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
    Write-Output 'server and keep-awake stopped'
    exit 0
}

function Start-KeepAwake {
    if ($NoKeepAwake) { return }

    $already = Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
        Where-Object { $_.CommandLine -like '*keep_awake_while_serving*' }
    if ($already) {
        Write-Output 'keep-awake is already running'
    } else {
        # Prefer the repo venv. A bare 'python' resolves to the WindowsApps
        # execution alias, which Start-Process launches into nothing -- the
        # launcher reported success and no process existed.
        $py = Join-Path $repo '.venv\Scripts\python.exe'
        if (-not (Test-Path $py)) {
            $py = (Get-Command python.exe -ErrorAction SilentlyContinue |
                   Where-Object { $_.Source -notlike '*WindowsApps*' } |
                   Select-Object -First 1).Source
        }
        if (-not $py) {
            Write-Output 'keep-awake SKIPPED: no usable python found (venv missing, only the Store alias on PATH)'
        } else {
            Start-Process -FilePath $py -WindowStyle Hidden -ArgumentList @(
                (Join-Path $repo 'scripts\inference\keep_awake_while_serving.py'),
                '--wait'
            )
            Start-Sleep -Seconds 2
            $running = Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
                Where-Object { $_.CommandLine -like '*keep_awake_while_serving*' }
            if ($running) {
                Write-Output "keep-awake holding the machine awake (pid $($running.ProcessId))"
            } else {
                Write-Output 'keep-awake FAILED to start; the machine may sleep'
            }
        }
    }
}

if (Test-Server) {
    Write-Output 'A server is already answering on 11435.'
    Write-Output 'Stop it first with -Stop if you want to replace it.'
    # Still ensure the reprieve: an already-running server that
    # nothing is holding awake is exactly the case this exists for.
    Start-KeepAwake
    exit 0
}

# The command must carry its own double quotes. Start-Process does not quote an
# argument that contains spaces, so wsl.exe received `bash -lc bash /home/...`
# as separate arguments, ran nothing, and exited 1 without writing to any log --
# which looks exactly like a server that failed to boot.
$command = '"bash /home/lukeb/brittain4/serve.sh >> /home/lukeb/brittain4/serve.log 2>&1"'
wsl bash -lc "echo '=== starting new server run ===' >> /home/lukeb/brittain4/serve.log"
Start-Process -FilePath 'wsl.exe' -WindowStyle Hidden -ArgumentList @('bash', '-lc', $command)
Write-Output 'server launching (detached); loading takes about 85 seconds'

# Wait for readiness, but watch for the process dying rather than only polling
# for success -- a waiter that knows only what success looks like cannot tell
# "still loading" from "never coming back".
$deadline = (Get-Date).AddMinutes(5)
while ((Get-Date) -lt $deadline) {
    Start-Sleep -Seconds 5
    if (Test-Server) {
        Write-Output 'server is up on http://localhost:11435'
        break
    }
    $alive = (wsl bash -lc 'pgrep -f "bin/vll""m serve" | head -1') -ne ''
    if (-not $alive) {
        Write-Output 'server process exited during startup. Last log lines:'
        wsl bash -lc 'tail -20 /home/lukeb/brittain4/serve.log'
        exit 1
    }
}
if (-not (Test-Server)) {
    Write-Output 'timed out waiting for the server. Last log lines:'
    wsl bash -lc 'tail -20 /home/lukeb/brittain4/serve.log'
    exit 1
}

Start-KeepAwake
