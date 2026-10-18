#include "reward_profiles.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
namespace {
void close(float a,float b,const char* label,float epsilon=.005f){if(!std::isfinite(a)||std::abs(a-b)>epsilon)throw std::runtime_error(std::string(label)+": "+std::to_string(a)+" vs "+std::to_string(b));}
RgFrame pose(float t,float score,float height=1){RgFrame f{};f.body_count=12;f.sim_time=t;f.score=score;f.left_foot_grounded=f.right_foot_grounded=1;
f.bodies[kTorsoBodySlot].x=score*400;f.bodies[kTorsoBodySlot].y=429.8f-height*506.85f;
f.bodies[kHeadBodySlot].x=score*400+150*std::sin(5*3.14159265f/180);f.bodies[kHeadBodySlot].y=f.bodies[kTorsoBodySlot].y-150*std::cos(5*3.14159265f/180);return f;}
float terminalDifference(float time,bool failed=false,bool truncated=false,bool sustainedLow=false,bool isolatedLow=false){
auto course=makeCourseSpec(GameMode::Classic100m);
// This isolated terminal-component fixture must not synthesize an instant
// 0-to-100m hurdle/recovery crossing; hurdle stages are independently tested.
course.hurdleLocations={500*400};
RewardState ending,ongoing;ending.reset(course);ongoing.reset(course);
auto begin=pose(0,0);ending.calculate(RewardProfile::Classic100m,course,begin,false);ongoing.calculate(RewardProfile::Classic100m,course,begin,false);
if(sustainedLow||isolatedLow){auto before=pose(time-1.0f/30,99,sustainedLow?.6f:1);
ending.calculate(RewardProfile::Classic100m,course,before,false);ongoing.calculate(RewardProfile::Classic100m,course,before,false);}
auto last=pose(time,100,isolatedLow?.6f:1);auto live=last;last.jump_landed=1;last.failed=failed?1:0;
return ending.calculate(RewardProfile::Classic100m,course,last,truncated)-ongoing.calculate(RewardProfile::Classic100m,course,live,false);}
}
int main(){try{
for(float t:{30.f,43.f,44.f,45.f,55.f,65.f,70.f}){
float expected=120-t+1200*std::clamp((65-t)/(65-43),0.f,1.f);
close(terminalDifference(t),expected,"qualified native jump_landed success has approved bounded time component");}
close(terminalDifference(44)-terminalDifference(45),1+1200.f/22,"success time value slope from 43 to 65 is 55.545 points per second");
close(terminalDifference(65)-terminalDifference(66),1,"outside fast-finish quality range only original 120-time slope remains");
close(terminalDifference(42)-terminalDifference(43),1,"faster than target43 fast bonus remains capped1200");
close(terminalDifference(45,true),-30,"failed terminal cannot collect success finish bonus");
close(terminalDifference(45,false,true),-30,"truncated terminal cannot collect success finish bonus");
close(terminalDifference(45,false,false,true),-30,"sustained low-posture fraction cannot collect success bonus");
// Preserve the original grace/fraction rule: isolated low final posture is
// not an extra instantaneous failure condition or a physics rule change.
close(terminalDifference(45,false,false,false,true),120-45+1200.f*20/22,"isolated low frame within original grace does not rewrite success rules");
std::cout<<"time_priority_reward_behavior_checks=passed\nsuccess_time_slope="<<1+1200.f/22<<"\n";return 0;
}catch(const std::exception& e){std::cerr<<"time_priority_reward_behavior_checks=failed: "<<e.what()<<"\n";return 1;}}
