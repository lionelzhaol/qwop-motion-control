#include "reward_profiles.h"
#include "early_start_baseline_reward.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
namespace {
constexpr float pi=3.14159265358979323846f;
constexpr float dt=1.0f/30;
void close(float a,float b,const char* label,float eps=2e-3f){if(!std::isfinite(a)||std::abs(a-b)>eps)throw std::runtime_error(std::string(label)+": "+std::to_string(a)+" vs "+std::to_string(b));}
void check(bool condition,const char* label){if(!condition)throw std::runtime_error(label);}
RgFrame frame(float t,float meters,float height=1,float lean=5){RgFrame f{};f.body_count=12;f.sim_time=t;f.score=meters;f.left_foot_grounded=f.right_foot_grounded=1;
f.bodies[kTorsoBodySlot].x=meters*400;f.bodies[kTorsoBodySlot].y=429.8f-506.85f*height;
f.bodies[kHeadBodySlot].x=meters*400+150*std::sin(lean*pi/180);f.bodies[kHeadBodySlot].y=f.bodies[kTorsoBodySlot].y-150*std::cos(lean*pi/180);return f;}
struct Pair{CourseSpec course=makeCourseSpec(GameMode::Classic100m);RewardState current;BaselineRewardState baseline;
Pair(){course.hurdleLocations={500*400};reset();}
void reset(){current.reset(course);baseline.reset(course);}
float step(const RgFrame& f,bool truncated=false){return current.calculate(RewardProfile::Classic100m,course,f,truncated)-baseline.calculate(RewardProfile::Classic100m,course,f,truncated);}};
float crossing(float t){Pair p;p.step(frame(0,0));return p.step(frame(t,3));}
}
int main(){try{
    const float fast=crossing(2),slow=crossing(3),deadline=crossing(4),late=crossing(5.5f);
    close(fast,15,"2s crossing receives 3 dense points plus 12 milestone points");
    close(slow,7.5f,"3s crossing receives 1.5 dense points plus 6 milestone points");
    check(fast>slow,"faster acceleration receives greater incremental reward");close(deadline,0,"4s deadline grants no new reward");close(late,0,"late progress grants no new reward");
    Pair segmented;segmented.step(frame(0,0));float total=0,expectedDense=0;
    for(int i=1;i<=90;++i){const float t=2.0f*i/90;total+=segmented.step(frame(t,3.0f*i/90));expectedDense+=2*(3.0f/90)*std::clamp((4-t)/4,0.0f,1.0f);}
    close(total,expectedDense+12,"segmented rollout receives only bounded new-distance reward and one milestone");
    check(total<=18.001f&&expectedDense<=6.001f,"dense six plus milestone twelve bound holds");
    close(segmented.step(frame(2+dt,3)),0,"no progress cannot earn new reward");
    close(segmented.step(frame(2+2*dt,2)),0,"backward movement cannot earn reward");
    close(segmented.step(frame(2+3*dt,3)),0,"return to recorded maximum cannot repeat either bonus");
    Pair stationary;stationary.step(frame(0,0));for(int i=1;i<=120;++i)close(stationary.step(frame(i*dt,0)),0,"stationary steps receive no early reward");
    Pair maxInstant;maxInstant.step(frame(0,0));close(maxInstant.step(frame(0,3)),18,"absolute 18 point cap even at zero elapsed time");
    for(int reason=1;reason<=4;++reason){Pair p;p.step(frame(0,0));auto f=frame(2,3,reason==2?.6f:1,reason==4?-20:5);if(reason==1)f.failed=1;
        close(p.step(f,reason==3),0,"failure low posture truncation or severe backward lean cannot collect first-cross bonuses");
        close(p.step(frame(2+dt,3)),0,"standing later cannot recover consumed early milestone or bank dense progress");
        close(p.step(frame(2+2*dt,2)),0,"disqualified crossing cannot be reset by backtracking");
        close(p.step(frame(2+3*dt,3)),0,"recross cannot recover disqualified milestone");
    }
    // Previous sustained crawling disqualifies the milestone even if the crossing itself is upright.
    Pair history;history.step(frame(0,0));history.step(frame(1,0,.6f));history.step(frame(1.5f,0,.6f));
    close(history.step(frame(2,3)),3,"excessive previous low posture earns only qualified new-distance dense feedback, never milestone");
    close(history.step(frame(2+dt,3)),0,"posture history milestone cannot be recovered later");
    segmented.reset();segmented.step(frame(0,0));close(segmented.step(frame(2,3)),15,"reset clears maximum progress and consumed early milestone");
    // Late 10m start bonuses remain unchanged; at score==10 the separately
    // approved v36 flat-speed term applies, so isolate its exact increment.
    for(float t:{5.5f,6.5f,7.5f}){Pair p;p.step(frame(0,0));const float speed=10/t;
        const float localSpeedIncrement=10*(8*std::clamp((speed-1.5f)/(2.8f-1.5f),0.0f,1.0f)-6*std::clamp(speed-1.5f,0.0f,1.0f));
        close(p.step(frame(t,10)),localSpeedIncrement,"late 10m differs only by approved flat-speed term, not early start bonus");}
    std::cout<<"early_start_reward_behavior_checks=passed\nfast_2s_increment="<<fast<<"\nslow_3s_increment="<<slow<<"\nsegmented_dense_reward="<<expectedDense<<"\nsegmented_total_increment="<<total<<"\n";return 0;
}catch(const std::exception& e){std::cerr<<"early_start_reward_behavior_checks=failed: "<<e.what()<<"\n";return 1;}}
