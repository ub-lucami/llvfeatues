# this powershell script will identify scenes in all files with mp4 extension and will write the results in 
# requirements: run in Python environment with scenedetect installed (pip install --upgrade scenedetect[opencv]) 
$oldvids = Get-ChildItem -Filter "*.mp4"
foreach ($oldvid in $oldvids) {
	scenedetect -i $oldvid.name detect-content -t 10  list-scenes
}
