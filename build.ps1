# Builds maya-ik2.exe. -Arch: gfx1100, gfx1101 or gfx1151 (several with ;). The ROCm SDK wheels live in -Venv.
param([string]$Arch = "gfx1100", [string]$Venv, [string]$Out)
$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path   # ($PSScriptRoot is empty in Windows PowerShell 5.1's defaults)
if (-not $Venv) { $Venv = Join-Path $here ".venv" }
if (-not $Out) { $Out = Join-Path $here "build" }
$rocm = "$Venv\Lib\site-packages\_rocm_sdk_devel"
if (-not (Test-Path "$rocm\lib\llvm\bin\clang++.exe")) { throw "no ROCm SDK in $rocm (README.md, Build)" }
$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
# (Visual Studio 2022 only: ROCm 7.14.1's compiler does not accept the headers of Visual Studio 18)
$vs = & $vswhere -products * -version "[17.0,18.0)" -latest -property installationPath
if (-not $vs -or -not (Test-Path "$vs\VC\Auxiliary\Build\vcvars64.bat")) { throw "Visual Studio 2022 C++ build tools not found" }
$bat = "$vs\VC\Auxiliary\Build\vcvars64.bat"
$r = $rocm.Replace('\', '/'); $v = $Venv.Replace('\', '/')
$gen = "`"$Venv\Scripts\cmake.exe`" -G Ninja -DCMAKE_MAKE_PROGRAM=`"$v/Scripts/ninja.exe`" -DCMAKE_C_COMPILER=`"$r/lib/llvm/bin/clang.exe`" -DCMAKE_CXX_COMPILER=`"$r/lib/llvm/bin/clang++.exe`" -DCMAKE_HIP_COMPILER=`"$r/lib/llvm/bin/clang++.exe`" -DCMAKE_HIP_COMPILER_ROCM_ROOT=`"$r`" -DCMAKE_PREFIX_PATH=`"$r`" -DCMAKE_BUILD_TYPE=Release `"-DCMAKE_HIP_ARCHITECTURES=$Arch`" -S `"$here`" -B `"$Out`""
$tmp = Join-Path $env:TEMP ("ik2build_" + [guid]::NewGuid().ToString("N") + ".bat")
"@echo off`r`ncall `"$bat`" >nul`r`n$gen || exit /b 3`r`n`"$Venv\Scripts\cmake.exe`" --build `"$Out`" || exit /b 4" | Set-Content $tmp -Encoding ascii
try { & cmd /c $tmp; $code = $LASTEXITCODE } finally { Remove-Item $tmp }
if ($code) { throw "build failed ($code)" }
"built $Out\maya-ik2.exe"
