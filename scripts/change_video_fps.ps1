# this powershell script will read all files with mp4 extension and will convert them to NTSC framerate (29.97fps)
# requirements: ffmpeg installed an in computer's PATH
# compatible with Windows 10 and Windows 11
$oldvids = Get-ChildItem -Filter "*.mp4"
foreach ($oldvid in $oldvids) {
	ffmpeg -i $oldvid.name -filter:v fps='ntsc' "$($oldvid.Directory)\30fps\$($oldvid.Name)"
}