# llvfeatues

A collection of scripts for low level video feature analysis regarding User Engagement

preparation of videos and calculation of low level features 

## Installation

Clone the repository.

Install [ffmpeg](https://ffmpeg.org/) (optional)

Install [pyscenedetect](https://scenedetect.com/en/latest/download/#download-and-installation) (preferably under your Python installation using pip)



## Usage

### framerate conversion 

```ps1
.\change_video_fps.ps1
```

This powershell script will read all files with mp4 extension and will convert them to NTSC framerate (29.97fps)

Requirements: [ffmpeg](https://ffmpeg.org/) installed an in computer's PATH

Script is compatible with Windows 10 and Windows 11

It is advisable to use a uniform frame rate before low level features are calculated. 
This will ensure a balanced interpretation of time-related video features.
We recommend to use uniform frame-rate sources to avoid possible artefacts induced by video conversion. 

### low level video feature extraction

We provide a Matlab set of scripts which analyze all video files in a single folder and provide a set of low-level video features in a csv file.

Before running the low level video feature extraction script, a python scene detection algorithm must be run. Execution of Powershell script in your Python environment:

```ps1
.\sc.ps1
``` 
will generate a file "FILENAME-Scenes.csv" for each of the *.mp4 files in the folder. 
Then, the necessary data for feature extraction are ready.  

Then, run the script "getVideoFeatures.m" in your Matlab environment. Expect the analysis to take a long time. The script requires Matlab function colorspace() to run properly. The script will generate files "FILENAME-T_Vec.csv" and "FILENAME-T_Avgs.csv" containing features per frame and the averages thereof for the entire video, respectively.

Function colorspace() courtesy of Pascal Getreuer (2022). Colorspace Transformations (https://www.mathworks.com/matlabcentral/fileexchange/28790-colorspace-transformations), MATLAB Central File Exchange. Retrieved October 10, 2022.

### hypothesis testing



```
## License
[MIT](https://choosealicense.com/licenses/mit/)