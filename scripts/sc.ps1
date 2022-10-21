# this powershell script will identify scenes in all files with mp4 extension and will write the results in FILENAME-Scenes.csv
# requirements: run in Python environment with scenedetect installed (pip install --upgrade scenedetect[opencv]) 
# compatible with Windows 10 and Windows 11
# usage:
# sc -folder "..\samples"
# default $folder =.
param (
    [string]$folder = "."
)
$oldvids = Get-ChildItem -Path $folder -Filter "*.mp4"
foreach ($oldvid in $oldvids) {
	scenedetect -i "$($oldvid.Directory)\$($oldvid.Name)" -o "$($oldvid.Directory)" detect-content -t 10  list-scenes
}
