# Run Call of Duty 3 in Xenia under RenderDoc, to capture frames as the game draws them (capture_dump.py
# then dumps the draws, capture_check.py compares them with the extracted models).
#   powershell -ExecutionPolicy Bypass -File xenia_capture.ps1 -Disc <extracted disc folder> [-Mode mp|sp]
#        [-Xenia <xenia_canary.exe>] [-RenderDoc <renderdoccmd.exe>] [-Out <captures folder>]
# In the game press F12 (RenderDoc's capture key). The disc path must not contain spaces (renderdoccmd
# splits arguments at spaces): use a directory junction (New-Item -ItemType Junction) if it does.
param(
    [Parameter(Mandatory = $true)][string]$Disc,
    [string]$Mode = 'mp',
    [string]$Xenia = 'xenia_canary.exe',
    [string]$RenderDoc = 'renderdoccmd.exe',
    [string]$Out = 'captures'
)
$xex = if ($Mode -eq 'sp') { Join-Path $Disc 'default.xex' } else { Join-Path $Disc 'codmp_xenonf.xex' }
New-Item -ItemType Directory -Force $Out | Out-Null
$xeniaDir = Split-Path -Parent (Resolve-Path $Xenia)
Write-Host "Starting $xex in Xenia under RenderDoc. Press F12 in the game to capture; captures -> $Out"
& $RenderDoc capture --wait-for-exit --working-dir $xeniaDir --capture-file (Join-Path (Resolve-Path $Out) 'cod3') $Xenia $xex
Get-ChildItem $Out -Filter *.rdc | Select-Object Name, Length, LastWriteTime | Format-Table
