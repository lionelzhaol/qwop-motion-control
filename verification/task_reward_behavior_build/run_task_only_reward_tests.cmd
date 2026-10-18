@echo off
call "E:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 exit /b %errorlevel%
cl /nologo /std:c++20 /EHsc /utf-8 /I"..\..\QWOP_rl_training_public\src" "..\test_task_only_reward.cpp" "..\..\QWOP_rl_training_public\src\reward_profiles.cpp" "..\..\QWOP_rl_training_public\src\course_rules.cpp" /Fe:task_only_reward_test.exe
if errorlevel 1 exit /b %errorlevel%
task_only_reward_test.exe
exit /b %errorlevel%
