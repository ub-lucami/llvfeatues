# this powershell script will run matlab script to get video features of all files with mp4 extension 
# and will write the results in files
# FILENAME--T_Vec.csv
# FILENAME--T_Avgs.csv
# compatible with Windows 10 and Windows 11
# usage:
# mergeLLFeatures -folder "..\samples" -tablename "LLFeatures.csv"
# default $folder =.
param (
    [string]$folder = ".",
    [string]$tablename = "LLFeatures.csv"
)
matlab -batch "mergeLLFeatures('$($folder)', '$($tablename)')"

