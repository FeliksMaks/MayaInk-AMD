# One greedy run of the engine on the GPU whose name contains -Gpu (token ids in, token ids out, and the speed).
param([Parameter(Mandatory)][string]$Model, [string]$Gpu = "7900", [int]$Tokens = 200, [int]$Context = 8192, [int]$Chunk = 4096,
      [string]$Prompt, [string]$Venv, [string]$Exe)
$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path   # ($PSScriptRoot is empty in Windows PowerShell 5.1's defaults)
if (-not $Prompt) { $Prompt = Join-Path $here "scripts\prompt27.txt" }
if (-not $Venv) { $Venv = Join-Path $here ".venv" }
if (-not $Exe) { $Exe = Join-Path $here "build\maya-ik2.exe" }
$env:PATH = "$Venv\Lib\site-packages\_rocm_sdk_devel\bin;$env:PATH"
Remove-Item Env:HIP_VISIBLE_DEVICES -ErrorAction SilentlyContinue
$idx = & "$Venv\Scripts\python.exe" "$here\scripts\hip_gpu_index.py" $Gpu
if ($LASTEXITCODE -ne 0) { throw "no GPU whose name contains '$Gpu'" }
$env:HIP_VISIBLE_DEVICES = "$idx"
$env:IK2_GATE_BF16 = "1"; $env:IK2_GATE_TILE = "64"; $env:IK2_UNSUPPORTED = "1"
& $Exe --model $Model --ctx $Context --chunk $Chunk --tokens "@$Prompt" --gen $Tokens
