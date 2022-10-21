function getVideoFeatures(folder)
% function getVideoFeatures(folder)
% low-level video metadata extraction 
%
% required: "colorspace.mexw64"
% A) video scene breakup
% before running the script, python script tor scene identification must be run.
% open ANACONDA powershell to identify scenes from the video
% call script "sc.ps1 " or manually extract scene timestampps using:
% scenedetect -i "PATH\FILENAME.mp4" detect-content -t 10  list-scenes
% the above command generates file "FILENAME-Scenes.csv"
% - first line contains timestamps of Scenes ENDPOINTS
% - follows a table with detailed data for each scene (in order as follows):
%   Scene Number,Start Frame,Start Timecode,Start Time (seconds),End Frame,...
%   End Timecode,End Time (seconds),Length (frames),Length (timecode),Length (seconds)
%
% output 
% FILENAME-T_Vec.csv
% frame-related values of "ShotNum" (scene ID) , "ColVar", "MotionMean",
% "MotionStd" and "LightKey"
%
% FILENAME-T_Avgs.csv
% average values of "avgShotLen", "ColVar", "MotionMean", "MotionStd" and "LightKey"
% for entire video
warning off;
if exist("folder")
    videoPath=strcat(folder,"\");
else
    videoPath='.\';
end
videoFileNames=dir(strcat(videoPath,'*.mp4'));
numVideo=length(videoFileNames);
for videoIdx=1:numVideo
        %opens video mp4 for reading
        vidObj = VideoReader(strcat(videoPath,videoFileNames(videoIdx).name));
        % read scene data
        [dir1,name,ext]=fileparts(strcat(videoPath,videoFileNames(videoIdx).name));
        % opens file *-scenes.csv for reading: analyse video file and get
        % SCENE averages.
        opts = detectImportOptions(strcat(dir1,'\',name,'-Scenes.csv'));
        opts.SelectedVariableNames = {'SceneNumber','StartFrame','EndFrame'};
        Scenes = readtable(strcat(dir1,'\',name,'-Scenes.csv'),opts);
        numFrames=max(Scenes.EndFrame+1);
        ShotNum = zeros(numFrames,1);
        ColVar = zeros(numFrames,1);
        MotionMean = zeros(numFrames,1);
        MotionStd = zeros(numFrames,1);
        LightKey = zeros(numFrames,1);
        for n=1:max(Scenes.SceneNumber)
            ShotNum(Scenes(n,:).StartFrame+1:Scenes(n,:).EndFrame+1)=Scenes(n,:).SceneNumber;
        end
        % prepare in advance for Optical Flow HS
        opticFlow = opticalFlowHS;
        fCount=0;
        %loop through frames for 
        textprogressbar(char(strcat(num2str(videoIdx),"/",num2str(numVideo)," ",name,": ")));
        while(hasFrame(vidObj))
            frameRGB = lin2rgb(double(readFrame(vidObj))/255);
            fCount=fCount+1;
            %waitbar(fCount/numFrames,wbar,name);
            textprogressbar(100*(fCount+1)/numFrames);
            % determine Color Variance
            frameLuv = colorspace('Luv<-',frameRGB);
            covLuv   = cov(reshape(frameLuv,[],3));
            ColVar(fCount)  = det(covLuv);
            % determine optical flow avg and std on magnitude
            flow = estimateFlow(opticFlow,im2gray(frameRGB));
            MotionMean(fCount) = mean2(flow.Magnitude);
            MotionStd(fCount) = std2(flow.Magnitude);
            % determine Lighting Key = hsValue.mean * hsValue.std
            frameHSV = rgb2hsv(frameRGB);
            LightKey(fCount) = mean2(frameHSV(:,:,3))*std2(frameHSV(:,:,3));
        end
        textprogressbar(' OK');
        %averages overall
        avgShotLen = fCount/height(Scenes);
        avgColVar = mean(ColVar);
        avgMotionMean = mean(MotionMean);
        avgMotionStd = mean(MotionStd);
        avgLightKey = mean(LightKey);
        
        %save results
        T_Vec = table(ShotNum, ColVar, MotionMean, MotionStd, LightKey);
        T_Avgs = table(avgShotLen, avgColVar, avgMotionMean, avgMotionStd, avgLightKey);
        writetable(T_Vec, strcat(videoPath,name,'-T_Vec.csv'));
        writetable(T_Avgs, strcat(videoPath,name,'-T_Avgs.csv'));
    disp(strcat('Done: ',name));
end


