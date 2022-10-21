# this powershell script will run matlab script to get video features of all files with mp4 extension 
# and will write the results in files
# FILENAME--T_Vec.csv
# FILENAME--T_Avgs.csv
# compatible with Windows 10 and Windows 11
# usage:
# getVideoFeatures -folder "..\samples"
# default $folder =.
param (
    [string]$folder = "."
)
matlab -batch "getVideoFeatures('$($folder)')"

