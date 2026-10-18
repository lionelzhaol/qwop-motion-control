"""Compile separate reward counterfactuals without rebuilding the live extension."""
from pathlib import Path
import hashlib
import json
import re

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent / "QWOP_rl_training_public"
BUILD = BASE / "posture_audit_v39_build"
BUILD.mkdir(exist_ok=True)
source = (ROOT / "src/reward_profiles.cpp").read_text(encoding="utf-8")
header = (ROOT / "src/reward_profiles.h").read_text(encoding="utf-8")
variants = {
    "Current": {},
    "NoLeanProgress": {"kForwardLeanProgressBonus": "0.0f"},
    "NoLeanPenalty": {"kLeanPosturePenaltyPerSecond": "0.0f"},
    "NoVertical": {"kVerticalMotionPenaltyPerSecond": "0.0f"},
    "NoCadence": {"kCadenceProgressBonus": "0.0f"},
    "NoExactLean": {"kForwardLeanProgressBonus": "0.0f", "kLeanPosturePenaltyPerSecond": "0.0f"},
}
classes = []
for name, changes in variants.items():
    class_name = "Audit" + name + "RewardState"
    stem = "audit_" + name.lower()
    classes.append((class_name, stem, name))
    changed = source
    for constant, value in changes.items():
        changed, count = re.subn(r"(constexpr float " + re.escape(constant) + r" = )[^;]+;", r"\g<1>" + value + ";", changed)
        assert count == 1, constant
    changed = changed.replace('#include "reward_profiles.h"', f'#include "{stem}.h"')
    changed = re.sub(r"\bRewardState\b", class_name, changed)
    class_header = re.sub(r"\bRewardState\b", class_name, header)
    (BUILD / (stem + ".cpp")).write_text(changed, encoding="utf-8")
    (BUILD / (stem + ".h")).write_text(class_header, encoding="utf-8")
wrapper = '\n'.join(f'#include "{stem}.h"' for _, stem, _ in classes) + '\n'
wrapper += 'struct AuditState {\n CourseSpec course=makeCourseSpec(GameMode::Classic100m);\n'
wrapper += ''.join(f' {class_name} state{index};\n' for index, (class_name, _, _) in enumerate(classes))
wrapper += f' {classes[0][0]} task;\n AuditState(){{'
wrapper += ''.join(f'state{index}.reset(course);' for index in range(len(classes))) + 'task.reset(course);}\n};\n'
wrapper += 'extern "C" {\n__declspec(dllexport) void* audit_create(){return new AuditState;}\n'
wrapper += '__declspec(dllexport) void audit_destroy(void* p){delete static_cast<AuditState*>(p);}\n'
wrapper += '__declspec(dllexport) void audit_reward(void* p,const RgFrame* f,unsigned char truncated,float* out){auto& s=*static_cast<AuditState*>(p);\n'
wrapper += ''.join(f'out[{index}]=s.state{index}.calculate(RewardProfile::Classic100m,s.course,*f,truncated!=0);\n' for index in range(len(classes)))
wrapper += f'out[{len(classes)}]=s.task.calculate(RewardProfile::Classic100mTask,s.course,*f,truncated!=0);\n}}\n}}\n'
(BUILD / "posture_audit_wrapper.cpp").write_text(wrapper, encoding="utf-8")
(BUILD / "audit_course_rules.cpp").write_bytes((ROOT / "src/course_rules.cpp").read_bytes())
lines = ['@echo off', 'call "E:\\Program Files\\Microsoft Visual Studio\\2022\\BuildTools\\VC\\Auxiliary\\Build\\vcvars64.bat" >nul',
         'if errorlevel 1 exit /b %errorlevel%',
         'cl /nologo /std:c++20 /EHsc /utf-8 /LD /I"..\\..\\QWOP_rl_training_public\\src" posture_audit_wrapper.cpp '
         + ' '.join(stem + '.cpp' for _, stem, _ in classes) + ' audit_course_rules.cpp /Fe:posture_audit_v39.dll',
         'exit /b %errorlevel%']
(BUILD / "run_posture_audit_build.cmd").write_bytes(('\r\n'.join(lines) + '\r\n').encode('ascii'))
(BUILD / "manifest.json").write_text(json.dumps({"original_reward_cpp_sha256": hashlib.sha256(source.encode('utf-8')).hexdigest(),
                                                 "variants": variants, "output_order": [name for _, _, name in classes] + ["TaskOnly"]}, indent=2), encoding="utf-8")
print("Prepared verification-only C++ reward variants")
