#include "reward_profiles.h"
#include "cadence_zero_reward.h"
struct AuditState {
    CourseSpec course=makeCourseSpec(GameMode::Classic100m);
    RewardState current;
    NoCadenceRewardState noCadence;
    AuditState(){current.reset(course);noCadence.reset(course);}
};
extern "C" {
__declspec(dllexport) void* audit_create(){return new AuditState;}
__declspec(dllexport) void audit_destroy(void* ptr){delete static_cast<AuditState*>(ptr);}
__declspec(dllexport) void audit_reward(void* ptr,const RgFrame* frame,unsigned char truncated,float* output){
    auto& state=*static_cast<AuditState*>(ptr);
    output[0]=state.current.calculate(RewardProfile::Classic100m,state.course,*frame,truncated!=0);
    output[1]=state.noCadence.calculate(RewardProfile::Classic100m,state.course,*frame,truncated!=0);
}
}
