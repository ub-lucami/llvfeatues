# llvfeatues

A collection of scripts for low level video feature analysis regarding User Engagement

preparation of videos and calculation of low level features 


## Installation

Clone the repository.

Install [ffmpeg](https://ffmpeg.org/) (optional)

Install [pyscenedetect](https://scenedetect.com/en/latest/download/#download-and-installation) (preferably under your Python installation using pip)


## Usage

Follow the steps below in the listed order to provide necessary data.

Video analysis steps may be omitted if results are provided in adequate data files.

### 1. Framerate Conversion 

It is advisable to use a uniform frame rate before low level features are calculated. 
This will ensure a balanced interpretation of time-related video features.
We recommend to use uniform frame-rate sources to avoid possible artefacts induced by video conversion. 

```ps1
.\change_video_fps.ps1 -folder FOLDERNAME  -subfolder SUBFOLDERNAME
```

This powershell script will read all files with mp4 extension and will convert them to NTSC framerate (29.97fps). FOLDERNAME indicates a specific folder location. Default folder location is  current folder. SUBFOLDERNAME indicates a subfolder which will be created under FOLDERNAME where converted files will be stored.

Requirements: [ffmpeg](https://ffmpeg.org/) installed in computer's PATH

Script is compatible with Windows 10 and Windows 11.

### 2. Low Level Video Feature Extraction

We provide a Matlab set of scripts which analyze all video files in a single folder and provide a set of low-level video features in a csv file.

Before running the low level video feature extraction script, a python scene detection algorithm must be run. Execution of Powershell script in your Python environment:

```ps1
.\sc.ps1 -folder FOLDERNAME
``` 
will generate a file "FILENAME-Scenes.csv" for each of the *.mp4 files in the folder. 
FOLDERNAME indicates a specific folder location. Default folder location is  current folder.
Then, the necessary data for feature extraction are ready.  

Then, call the Matlab function "getVideoFeatures.m" in your Matlab environment. This can also be initiated from PowerShell using command 
```ps1
.\getVideoFeatures.ps1 -folder FOLDERNAME
```
Expect the analysis to take a long time. The script requires Matlab function colorspace() to run properly. The script will generate files "FILENAME-T_Vec.csv" and "FILENAME-T_Avgs.csv" containing features per frame and the averages thereof for the entire video, respectively.

Finally, call the Matlab function "mergeLLFeatures.m" in your Matlab environment. This can also be initiated from PowerShell using command 
```ps1
.\mergeLLFeatures.ps1 -folder FOLDERNAME -tablename FILENAME
```

Function colorspace() courtesy of Pascal Getreuer (2022). Colorspace Transformations (https://www.mathworks.com/matlabcentral/fileexchange/28790-colorspace-transformations), MATLAB Central File Exchange. Retrieved October 10, 2022.

### 3. Merge Scores and Features

Run the file "aggregateTables.py" to merge Low Level Video Features and User Engagement Scores into a single table.

Requirements:

pandas

Before the run specify folder and data filenames in script header, eg.:

(Cell 3) Features

```py
file_path = '../samples/'
data_fn = 'LLFeatures.csv'
```

(Cell 4) Scores

```py
data_fn = 'OBrienUES.csv'
```

(Cell 6)

```py
UES_LLF_fn = 'UES_LLFeatures.csv'
```
After edits, run script:
```py
.\landscape_feats_inbar.py
```

### 4. Statistics of UES-SF Scores and Low-Level Video Features

Run the file "landscape_feats_inbar.py" to get statistical data on Low Level features and User Engagement scores. 

Requirements:

numpy, pandas, matplotlib.pyplot, seaborn 

Before the run specify folder and data filename in script header (Cell 3), eg.:

```py
file_path = '../samples/'
data_fn = 'UES_LLFeatures.csv'
```
After edits, run script:
```py
.\landscape_feats_inbar.py
```

### 5. Hypothesis Testing

Run the file "runExperiment_ExplVar.py" to process all pairs of Low Level Video Features against User Engagement Scores. 

Requirements:

os, numpy, pandas

explVar_tools - provided in this library

Before the run specify folder and data filename in script header (Cell 3), eg.:

```py
file_path = '../samples/'
data_fn = 'UES_LLFeatures.csv'
```

After edits, run script:

```py
.\runExperiment_ExplVar.py
```

Please ignore irrelevant run-time warnings.

### 6. Maximum possible R2

As explained in our study, it is not the intention to build a complete regression model but to study only the contribution of video aspects towards the engagement scores. Hereby we use categorical video regression to have an insight into maximum acheivable R-squared values based on advertisement video only.
Variability of scores within a single video regardless of the features turns out to be high, which lead to low R-squared values.

Run the file "runExperiment_ExplVarIdeal.py" to determine maximum possible R-squared values for the case of ideal video features.

Requirements:

os, pandas

explVar_tools_4ideal - provided in this library

Before the run specify folder and data filename in script header (Cell 3), eg.:

```py
file_path = '../samples/'
data_fn = 'UES_LLFeatures.csv'
```

After edits, run script:

```py
.\runExperiment_ExplVarIdeal.py
```
## 7. Sample data

The required sample data are available in subfolder "./samples".
Low level features were obtained from video files available on YouTube.
A detailed list of files is provided with the dataset. 

Scores from crowdsourcing study are collected in file "OBrienUES.csv".

Video Feature values are provided in file "LLFeatures.csv" for convenience.
The results in this table were obtained from video files following the Steps 1-2 of this manual. 
By following Steps 1-2 one can perform video analysis of any given recordings.
Slight variations can arise due to numerical errors and format conversion.

## License
[MIT](https://choosealicense.com/licenses/mit/)