@echo off
setlocal enabledelayedexpansion

if exist list_of_csv_files.txt del list_of_csv_files.txt

set APK_DIR_BENIGN=samples\benign
set OUT_DIR_BENIGN=out\benign

if not exist "%OUT_DIR_BENIGN%" (
    mkdir "%OUT_DIR_BENIGN%"
)

for %%f in (%APK_DIR_BENIGN%\*.apk) do (
    echo 🔵 Processing benign: %%~nxf
    java -Xms512m -Xmx2048m -cp .;DroidASAT.jar;lib\rt.jar;lib\sootclasses-trunk-jar-with-dependencies.jar;lib\soot-infoflow.jar;lib\soot-infoflow-android.jar DroidASAT.main lib\rt.jar lib\android-jar "%%f" "%OUT_DIR_BENIGN%"
    echo ./out/benign/%%~nxf.csv >> list_of_csv_files.txt
)

set APK_DIR_ADWARE=samples\malware\adware
set OUT_DIR_ADWARE=out\malware\adware

if not exist "%OUT_DIR_ADWARE%" (
    mkdir "%OUT_DIR_ADWARE%"
)

for %%f in (%APK_DIR_ADWARE%\*.apk) do (
    echo 🔴 Processing malware/adware: %%~nxf
    java -Xms512m -Xmx2048m -cp .;DroidASAT.jar;lib\rt.jar;lib\sootclasses-trunk-jar-with-dependencies.jar;lib\soot-infoflow.jar;lib\soot-infoflow-android.jar DroidASAT.main lib\rt.jar lib\android-jar "%%f" "%OUT_DIR_ADWARE%"
    echo ./out/malware/adware/%%~nxf.csv >> list_of_csv_files.txt
)

echo ----------------------------
echo ✅ Finished processing benign and adware APKs
pause
endlocal
