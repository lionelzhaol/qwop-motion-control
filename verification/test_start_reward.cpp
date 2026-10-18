#include "reward_profiles.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

void expect(float actual, float expected, const char* name) {
    if (!std::isfinite(actual) || std::abs(actual-expected)>1e-3f)
        throw std::runtime_error(std::string(name)+": actual="+std::to_string(actual)+", expected="+std::to_string(expected));
}
RgFrame pose() {
    RgFrame f{};
    f.body_count=RG_PHYSICS_BODY_COUNT;
    f.bodies[kTorsoBodySlot].y=-77.05f;
    f.bodies[kHeadBodySlot].x=std::sin(5.0f*3.14159265f/180)*150;
    f.bodies[kHeadBodySlot].y=-77.05f-std::cos(5.0f*3.14159265f/180)*150;
    f.left_foot_grounded=1;
    return f;
}
int main() { try {
    const auto course=makeCourseSpec(GameMode::Classic100m);
    RewardState s;
    for(float t:{5.5f,6.5f,7.5f}) {
        s.reset(course);auto f=pose();s.calculate(RewardProfile::Classic100m,course,f,false);
        f.score=10;f.sim_time=t;
        f.bodies[kTorsoBodySlot].x=4000;f.bodies[kHeadBodySlot].x+=4000;
        const float v=10/t;
        // score==10 belongs to the approved flat-running reward window;
        // the pace deficit keeps the original 2.5m/s target below.
        const float speedBonus=80*std::clamp((v-1.5f)/(2.8f-1.5f),0.0f,1.0f);
        const float deficit=std::max(0.0f,1-v/2.5f);
        const float milestone=t==5.5f?40:t==6.5f?20:0;
        const float denseStart=40*std::clamp((9-t)/3.5f,0.0f,1.0f);
        expect(s.calculate(RewardProfile::Classic100m,course,f,false),34+speedBonus+0.2f*t-16*t*deficit*deficit+milestone+denseStart,"timed and dense 10m bonus graded and bounded");
        f.sim_time+=1.0f/30;
        expect(s.calculate(RewardProfile::Classic100m,course,f,false),(0.2f-16*deficit*deficit)/30,"holding 10m cannot claim bonus again");
    }
    s.reset(course);auto f=pose();s.calculate(RewardProfile::Classic100m,course,f,false);
    f.score=10;f.sim_time=5.5f;f.bodies[kTorsoBodySlot].x=4000;f.bodies[kHeadBodySlot].x+=4000;
    f.bodies[kTorsoBodySlot].y=150;
    const float deficit=1-(10/5.5f)/2.5f;
    const float height=(429.8f-150)/506.85f;
    expect(s.calculate(RewardProfile::Classic100m,course,f,false),5.5f*(height-0.8f)-0.2f*5.3f-16*5.5f*deficit*deficit,"crawling cannot collect start prize");
    f.bodies[kTorsoBodySlot].y=-77.05f;f.sim_time+=1.0f/30;
    expect(s.calculate(RewardProfile::Classic100m,course,f,false),(0.2f-16*deficit*deficit-4.0f)/30,"standing later cannot recover consumed prize and abrupt bobbing is penalized");
    s.reset(course);f=pose();s.calculate(RewardProfile::Classic100m,course,f,false);
    f.score=10;f.sim_time=5.5f;f.bodies[kTorsoBodySlot].x=4000;f.bodies[kHeadBodySlot].x+=4000;f.failed=1;
    const float speedBonus=80*std::clamp(((10/5.5f)-1.5f)/(2.8f-1.5f),0.0f,1.0f);
    expect(s.calculate(RewardProfile::Classic100m,course,f,false),34+speedBonus+1.1f-16*5.5f*deficit*deficit-30,"failed crossing cannot collect start prize");
    RewardState targetLean, excessiveLean; targetLean.reset(course); excessiveLean.reset(course);
    auto aligned=pose(), tilted=pose();
    tilted.bodies[kHeadBodySlot].x=std::sin(25.0f*3.14159265f/180)*150;
    tilted.bodies[kHeadBodySlot].y=-77.05f-std::cos(25.0f*3.14159265f/180)*150;
    targetLean.calculate(RewardProfile::Classic100m,course,aligned,false);
    excessiveLean.calculate(RewardProfile::Classic100m,course,tilted,false);
    aligned.sim_time=tilted.sim_time=3; aligned.score=tilted.score=7.5f;
    aligned.bodies[kTorsoBodySlot].x=tilted.bodies[kTorsoBodySlot].x=3000;
    aligned.bodies[kHeadBodySlot].x+=3000; tilted.bodies[kHeadBodySlot].x+=3000;
    if(!(targetLean.calculate(RewardProfile::Classic100m,course,aligned,false)
         >excessiveLean.calculate(RewardProfile::Classic100m,course,tilted,false)+10))
        throw std::runtime_error("soft lean feedback must prefer 5 degrees to excessive tilt");
    // Identical forward motion: only vertically oscillating runner is penalized.
    RewardState flat, bobbing, smallBobbing; flat.reset(course); bobbing.reset(course); smallBobbing.reset(course);
    auto smooth=pose(); flat.calculate(RewardProfile::Classic100m,course,smooth,false);
    bobbing.calculate(RewardProfile::Classic100m,course,smooth,false);
    smallBobbing.calculate(RewardProfile::Classic100m,course,smooth,false);
    float flatTotal=0,bobTotal=0,smallTotal=0;
    for(int i=1;i<=90;++i) {
        smooth.sim_time=i/30.0f;smooth.score=2.5f*smooth.sim_time;
        smooth.bodies[kTorsoBodySlot].x=smooth.score*400;
        smooth.bodies[kHeadBodySlot].x=smooth.bodies[kTorsoBodySlot].x+std::sin(5.0f*3.14159265f/180)*150;
        auto noisy=smooth;
        const float offset=i%2?40.0f:-40.0f;
        noisy.bodies[kTorsoBodySlot].y+=offset; noisy.bodies[kHeadBodySlot].y+=offset;
        flatTotal+=flat.calculate(RewardProfile::Classic100m,course,smooth,false);
        bobTotal+=bobbing.calculate(RewardProfile::Classic100m,course,noisy,false);
        auto small=smooth;
        small.bodies[kTorsoBodySlot].y+=i%2?5.0f:-5.0f;
        small.bodies[kHeadBodySlot].y+=i%2?5.0f:-5.0f;
        smallTotal+=smallBobbing.calculate(RewardProfile::Classic100m,course,small,false);
    }
    if(!(flatTotal>bobTotal+1.5f)) throw std::runtime_error("bobbing penalty must prefer smooth motion");
    if(std::abs(flatTotal-smallTotal)>0.05f) throw std::runtime_error("small fast oscillations must stay inside height grace");
    // Paired trajectories: debounce and consume contacts, reward only moving steps.
    auto cadenceDifference=[&](bool moving,bool jitter,bool failed) {
        RewardState alternating, fixed; alternating.reset(course); fixed.reset(course);
        auto base=pose(); base.left_foot_grounded=base.right_foot_grounded=1;
        alternating.calculate(RewardProfile::Classic100m,course,base,false);
        fixed.calculate(RewardProfile::Classic100m,course,base,false);
        float difference=0;
        for(int i=1;i<=180;++i) {
            base.sim_time=i/30.0f; base.score=moving?2.5f*base.sim_time:0;
            base.bodies[kTorsoBodySlot].x=base.score*400;
            base.bodies[kHeadBodySlot].x=base.bodies[kTorsoBodySlot].x+std::sin(5.0f*3.14159265f/180)*150;
            base.failed=failed?1:0;
            auto stepping=base;
            const int period=jitter?2:18;
            const int phase=(i-1)%period;
            const int side=((i-1)/period)%2;
            if(phase<(jitter?1:8)) {
                if(side==0) stepping.left_foot_grounded=0;
                else stepping.right_foot_grounded=0;
            }
            difference+=alternating.calculate(RewardProfile::Classic100m,course,stepping,false)
                -fixed.calculate(RewardProfile::Classic100m,course,base,false);
        }
        return difference;
    };
    // Preserve the same minimum weighted-step evidence after the approved
    // cadence coefficient changed from 2.4 to 0.3 (one eighth).
    if(!(cadenceDifference(true,false,false)>5*(0.3f/2.4f))) throw std::runtime_error("moving alternating steps must earn cadence feedback");
    expect(cadenceDifference(false,false,false),0,"stationary stomping cannot earn cadence feedback");
    expect(cadenceDifference(true,true,false),0,"contact jitter cannot earn cadence feedback");
    expect(cadenceDifference(true,false,true),0,"failed runner cannot earn cadence feedback");
    std::cout<<"start_reward_behavior_checks=passed\n";
    return 0;
} catch(const std::exception& e) {
    std::cerr<<"start_reward_behavior_checks=failed: "<<e.what()<<"\n";
    return 1;
}
}
