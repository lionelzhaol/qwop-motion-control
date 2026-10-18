#include "audit_current.h"
#include "audit_noleanprogress.h"
#include "audit_noleanpenalty.h"
#include "audit_novertical.h"
#include "audit_nocadence.h"
#include "audit_noexactlean.h"
struct AuditState {
 CourseSpec course=makeCourseSpec(GameMode::Classic100m);
 AuditCurrentRewardState state0;
 AuditNoLeanProgressRewardState state1;
 AuditNoLeanPenaltyRewardState state2;
 AuditNoVerticalRewardState state3;
 AuditNoCadenceRewardState state4;
 AuditNoExactLeanRewardState state5;
 AuditCurrentRewardState task;
 AuditState(){state0.reset(course);state1.reset(course);state2.reset(course);state3.reset(course);state4.reset(course);state5.reset(course);task.reset(course);}
};
extern "C" {
__declspec(dllexport) void* audit_create(){return new AuditState;}
__declspec(dllexport) void audit_destroy(void* p){delete static_cast<AuditState*>(p);}
__declspec(dllexport) void audit_reward(void* p,const RgFrame* f,unsigned char truncated,float* out){auto& s=*static_cast<AuditState*>(p);
out[0]=s.state0.calculate(RewardProfile::Classic100m,s.course,*f,truncated!=0);
out[1]=s.state1.calculate(RewardProfile::Classic100m,s.course,*f,truncated!=0);
out[2]=s.state2.calculate(RewardProfile::Classic100m,s.course,*f,truncated!=0);
out[3]=s.state3.calculate(RewardProfile::Classic100m,s.course,*f,truncated!=0);
out[4]=s.state4.calculate(RewardProfile::Classic100m,s.course,*f,truncated!=0);
out[5]=s.state5.calculate(RewardProfile::Classic100m,s.course,*f,truncated!=0);
out[6]=s.task.calculate(RewardProfile::Classic100mTask,s.course,*f,truncated!=0);
}
}
