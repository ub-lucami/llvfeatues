function mergeLLFeatures(folder, mergeFilename)
% function mergeLLFeatures(folder, mergeFilename)
% merge script combines low-level feature averages from individual videos into a single table.  
%
% default folder is ".\"
% default output mergeFilename is "LLFeatures.csv" within folder
% each row contains variables "videoID", "ShotNum" (scene ID) , "ColVar", "MotionMean",
% "MotionStd" and "LightKey"
mFilename='LLFeatures.csv';
if nargin>0
    videoPath=strcat(folder,"\");
    if nargin==2
        mFilename=mergeFilename;
    end
else
    videoPath='.\';
end
videoFileNames=dir(strcat(videoPath,'*-T_Avgs.csv'));
numVideo=length(videoFileNames);
LLFeatures=table;
for videoIdx=1:numVideo
    %opens feature table FILENAME-T_Avgs.csv for reading
    [dir1,name1,ext]=fileparts(strcat(videoPath,videoFileNames(videoIdx).name));
    opts = detectImportOptions(strcat(dir1,'\',name1,'.csv'));
    % for the case of segmented video, this part extracts the Ad
    % related features from sources. In case the videos were pure ads, a
    % total feature vector is taken and converted to *Ad feature names for
    % compatibility with data analysis algorithms
    try
        % Ad related features present
        opts.SelectedVariableNames = {'avgShotLenAd','avgColVarAd','avgMotionMeanAd','avgMotionStdAd','avgLightKeyAd'};
    catch
        % Assuming the entire video is an Ad; ONLY take TOTAL video features...
        opts.SelectedVariableNames = {'avgShotLen','avgColVar','avgMotionMean','avgMotionStd','avgLightKey'};
    end
    LLFeature = readtable(strcat(dir1,'\',name1,'.csv'),opts);
    try
        % ...and take TOTAL video features as Ad related in case there was
        % no Ad segmentation provided
        LLFeature = renamevars(LLFeature,["avgShotLen","avgColVar","avgMotionMean","avgMotionStd","avgLightKey"], ...
                                         ["avgShotLenAd","avgColVarAd","avgMotionMeanAd","avgMotionStdAd","avgLightKeyAd"]);
    end
    
    LLFeature.videoName = erase(name1,'-T_Avgs');
    LLFeatures=[LLFeatures;LLFeature];
end
% write a csv table for the entire video dataset in a folder
writetable(LLFeatures, strcat(videoPath,mFilename));

