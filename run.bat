@echo off
setlocal

:: 1) Class-path: tool + every jar in lib\
set "CP=.;DroidASAT.jar;lib/*"

:: 2) Main entry
set "MAIN_CLASS=DroidASAT.main"

:: 3) Paths to the Java RT jar and Android jar directory
set "RT_JAR=lib\rt.jar"
set "ANDROID_JAR_DIR=lib\android-jar"

:: 4) Ensure output dirs exist
if not exist out\benign  mkdir out\benign
if not exist out\malware mkdir out\malware

echo === Processing BENIGN APKs ===
for %%F in (samples\benign\*.apk) do (
    echo Processing: %%~nxF
    java -Xmx1G -cp "%CP%" %MAIN_CLASS% ^
         "%RT_JAR%" "%ANDROID_JAR_DIR%" "%%~fF" "out\benign"
)

echo === Processing MALWARE APKs ===
for /R samples\malware %%F in (*.apk) do (
    echo Processing: %%~nxF
    java -Xmx1G -cp "%CP%" %MAIN_CLASS% ^
         "%RT_JAR%" "%ANDROID_JAR_DIR%" "%%~fF" "out\malware"
)

echo === All done! ===
pause
endlocal
