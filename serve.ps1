# Serves the model through the upstream repository's OpenAI / Anthropic compatible server with this repository's engine.
#   .\serve.ps1 -Upstream C:\src\maya-inkling-mini -Model C:\models\Inkling-2-Mini-UD-IQ4_XS.gguf [-Gpu 7900] [-Context 49152]
# -Upstream: a checkout of alesha-pro/maya-inkling-mini at commit c520fd1 (only its Python server and tokenizer tool are used).
# -Context: tokens of one chat. On a 20 GB card: up to 65536 with the FP16 cache, up to 131072 with -Fp8Kv.
# -Chunk: prompt tokens read at once; 0 picks 4096 up to a 48K context, 3072 up to 64K, 2048 above (a full card gets slow).
param([Parameter(Mandatory)][string]$Upstream, [Parameter(Mandatory)][string]$Model, [string]$Gpu = "7900",
      [int]$Context = 49152, [int]$Chunk = 0, [switch]$Fp8Kv, [int]$Port = 18000, [string]$Venv, [string]$Exe)
$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
if (-not $Venv) { $Venv = Join-Path $here ".venv" }
if (-not $Exe) { $Exe = Join-Path $here "build\maya-ik2.exe" }
$py = Join-Path $Venv "Scripts\python.exe"
foreach ($p in @($Exe, $Model, $py, (Join-Path $Upstream "serve\server.py"))) { if (-not (Test-Path $p)) { throw "not found: $p" } }
if ($Chunk -le 0) { $Chunk = if ($Gpu -match "8060" -or $Context -le 49152) { 4096 } elseif ($Context -le 65536) { 3072 } else { 2048 } }

$env:PATH = "$Venv\Lib\site-packages\_rocm_sdk_devel\bin;$env:PATH"
Remove-Item Env:HIP_VISIBLE_DEVICES -ErrorAction SilentlyContinue
$idx = & $py "$here\scripts\hip_gpu_index.py" $Gpu
if ($LASTEXITCODE -ne 0) { throw "no GPU whose name contains '$Gpu'" }

$tok = Join-Path $here "build\tokenizer"
if (-not (Test-Path "$tok\tokenizer.json")) {
    "Writing the tokenizer files (once) ..."
    & $py (Join-Path $Upstream "tools\strata_tokenizer.py") --gguf $Model --out (Join-Path $here "build") `
        --chat-template (Join-Path $Upstream "tools\ik2\chat_template.jinja") --check
    if ($LASTEXITCODE -ne 0) { throw "the tokenizer tool failed" }
}
$cfg = @{
    exe = (Resolve-Path $Exe).Path; cwd = $here; tokenizer = $tok; model_name = "inkling-2-mini"
    args = @("--model", $Model, "--max-context", "$Context", "--chunk", "$Chunk")
    env = @{ IK2_GATE_BF16 = "1"; IK2_GATE_TILE = "64"; HIP_VISIBLE_DEVICES = "$idx" }
}
if ($Fp8Kv) { $cfg.env.IK2_KV = "fp8" }
$cfgPath = Join-Path $here "build\serve-config.json"
$cfg | ConvertTo-Json -Depth 5 | Set-Content -Path $cfgPath -Encoding utf8
"GPU $Gpu (HIP device $idx), context $Context, chunk $Chunk - http://127.0.0.1:$Port/v1, model inkling-2-mini"
Push-Location $Upstream
try { & $py serve\server.py --engine maya --config $cfgPath --host 127.0.0.1 --port $Port }
finally { Pop-Location }
