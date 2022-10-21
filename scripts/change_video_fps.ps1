# this powershell script will read all files with mp4 extension and will convert them to NTSC framerate (29.97fps)
# requirements: ffmpeg installed an in computer's PATH
# compatible with Windows 10 and Windows 11
# usage:
# change_video_fps -folder "..\samples" -subfolder "30fps"
# default $folder =.
# default $subfolder =30fps

param (
    [string]$folder = ".",
	[string]$subfolder = "30fps"
)
$oldvids = Get-ChildItem -Path $folder -Filter "*.mp4"
If(!(test-path -PathType container "$($oldvids[0].Directory)\$($subfolder)"))
{
	  New-Item -ItemType Directory -Path "$($oldvids[0].Directory)\$($subfolder)"
}
foreach ($oldvid in $oldvids) {
	ffmpeg -i "$($oldvid.Directory)\$($oldvid.Name)" -filter:v fps='ntsc' "$($oldvid.Directory)\$($subfolder)\$($oldvid.Name)"
	#echo ""
	#echo "$($oldvid.Directory)\$($subfolder)\$($oldvid.Name)"
}