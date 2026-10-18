@echo off
call "E:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 exit /b %errorlevel%
cl /nologo /std:c++20 /EHsc /I"..\QWOP_rl_training_public\src" test_flat_speed_reward.cpp flat_speed_baseline_reward.cpp "..\QWOP_rl_training_public\src\reward_profiles.cpp" "..\QWOP_rl_training_public\src\course_rules.cpp" /Fe:flat_speed_reward_test.exe
if errorlevel 1 exit /b %errorlevel%
flat_speed_reward_test.exe
exit /b %errorlevel%
