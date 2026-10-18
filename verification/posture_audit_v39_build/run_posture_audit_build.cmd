@echo off
call "E:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 exit /b %errorlevel%
cl /nologo /std:c++20 /EHsc /utf-8 /LD /I"..\..\QWOP_rl_training_public\src" posture_audit_wrapper.cpp audit_current.cpp audit_noleanprogress.cpp audit_noleanpenalty.cpp audit_novertical.cpp audit_nocadence.cpp audit_noexactlean.cpp audit_course_rules.cpp /Fe:posture_audit_v39.dll
exit /b %errorlevel%
