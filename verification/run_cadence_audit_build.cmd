@echo off
call "E:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 exit /b %errorlevel%
cl /nologo /std:c++20 /EHsc /LD /I"..\QWOP_rl_training_public\src" cadence_reward_audit_wrapper.cpp cadence_zero_reward.cpp "..\QWOP_rl_training_public\src\reward_profiles.cpp" "..\QWOP_rl_training_public\src\course_rules.cpp" /Fe:cadence_reward_audit.dll
exit /b %errorlevel%
