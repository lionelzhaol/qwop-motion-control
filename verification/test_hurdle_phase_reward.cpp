#include "reward_profiles.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {
constexpr float pi=3.14159265358979323846f;
constexpr float dt=1.0f/30.0f;
void close(float a,float b,const char* label,float eps=2e-3f) {
    if(!std::isfinite(a)||!std::isfinite(b)||std::abs(a-b)>eps)
        throw std::runtime_error(std::string(label)+": "+std::to_string(a)+" vs "+std::to_string(b));
}
void check(bool condition,const char* label) {if(!condition) throw std::runtime_error(label);}
RgFrame frame(float t,float meters,float height=1.0f,bool supported=true,float lean=5.0f) {
    RgFrame f{};f.body_count=RG_PHYSICS_BODY_COUNT;f.sim_time=t;f.score=meters;
    f.hurdle_location=50*400;f.bodies[kTorsoBodySlot].x=meters*400;
    f.bodies[kTorsoBodySlot].y=429.8f-height*506.85f;
    f.bodies[kHeadBodySlot].x=meters*400+std::sin(lean*pi/180)*150;
    f.bodies[kHeadBodySlot].y=f.bodies[kTorsoBodySlot].y-std::cos(lean*pi/180)*150;
    f.left_foot_grounded=f.right_foot_grounded=supported?1:0;return f;
}
struct Pair {
    CourseSpec near=makeCourseSpec(GameMode::Classic100m),far=near;
    RewardState a,b;
    Pair(){far.hurdleLocations={500*400};reset();}
    void reset(){a.reset(near);b.reset(far);}
    float step(const RgFrame& f,bool truncated=false){return a.calculate(RewardProfile::Classic100m,near,f,truncated)-b.calculate(RewardProfile::Classic100m,far,f,truncated);}
};
float recovery(float duration,int disqualified=0,bool recross=false,Pair* reuse=nullptr) {
    Pair local;Pair& p=reuse?*reuse:local;p.reset();
    p.step(frame(19,48));p.step(frame(20,50));
    const int n=static_cast<int>(std::round(duration*30));float total=0;
    for(int i=1;i<=n;++i) {
        auto f=frame(20+duration*i/n,50+8.0f*i/n,disqualified==2?.6f:1.0f,disqualified!=3);
        if(i==n && disqualified==1) f.failed=1;
        total+=p.step(f,disqualified==4&&i==n);
    }
    if(recross){close(p.step(frame(20+duration+dt,57)),0,"backtracking cannot claim bonus");close(p.step(frame(20+duration+2*dt,58)),0,"recovery cannot be reclaimed or recovered after disqualification");}
    return total;
}
}
int main(){try{
    Pair plain;
    for(int i=0;i<=600;++i) close(plain.step(frame(i*dt,2.5f*i*dt)),0,"ordinary running far before hurdle is unchanged");
    Pair relief;relief.step(frame(20,49));float reliefTotal=0;
    for(int i=1;i<=12;++i) reliefTotal+=relief.step(frame(20+i*dt,49+.08f*i,1.12f,false,25));
    check(reliefTotal>.5f,"nearby confirmed elevated flight must reduce posture and vertical penalties");
    Pair lowFlight;lowFlight.step(frame(20,49));
    for(int i=1;i<=12;++i) close(lowFlight.step(frame(20+i*dt,49+.08f*i,1.0f,false,25)),0,"ordinary low flight is not a hurdle jump");
    Pair deadline;deadline.step(frame(20,49,1.1f,true,25));
    bool seen=false;
    for(int i=1;i<=70;++i){float delta=deadline.step(frame(20+i*dt,51,1.1f,false,25));
        if(i>=4&&i<=50){check(delta>0,"valid jump must have a limited exemption");seen=true;}
        if(i>=61) close(delta,0,"jump exemption expires after 1.8 seconds");}
    check(seen,"jump confirmation exercised");
    for(int i=71;i<=80;++i) close(deadline.step(frame(20+i*dt,50.5f,1.1f,false,25)),0,"backtracking cannot reopen expired jump");
    deadline.reset();deadline.step(frame(20,49,1.1f,true,25));
    for(int i=1;i<=5;++i){float delta=deadline.step(frame(20+i*dt,51,1.1f,false,25));
        if(i>=4)check(delta>0,"reset clears closed jump state");}
    RewardState stationary;auto stationaryCourse=makeCourseSpec(GameMode::Classic100m);stationary.reset(stationaryCourse);
    stationary.calculate(RewardProfile::Classic100m,stationaryCourse,frame(0,51,1.0f,true,25),false);
    float stationaryReward=0;
    for(int i=1;i<=90;++i)stationaryReward+=stationary.calculate(RewardProfile::Classic100m,stationaryCourse,frame(i*dt,51,1.12f,false,25),false);
    check(stationaryReward<0,"stationary hurdle hopping cannot earn positive total reward");
    Pair spatial;spatial.step(frame(20,49,1.1f,true,25));
    for(int i=1;i<=5;++i) spatial.step(frame(20+i*dt,51,1.1f,false,25));
    close(spatial.step(frame(20+6*dt,54,1.1f,false,25)),0,"jump exemption closes four meters after hurdle");
    close(spatial.step(frame(20+7*dt,51,1.1f,false,25)),0,"backtracking cannot reopen distance-closed jump");
    // A moving or knocked hurdle location cannot move the fixed reward window.
    Pair moved;moved.step(frame(20,49,1.1f,true,25));
    for(int i=1;i<=5;++i){auto f=frame(20+i*dt,51,1.1f,false,25);f.hurdle_location=500*400;
        float delta=moved.step(f);if(i>=4)check(delta>0,"dynamic hurdle coordinate must not shift fixed phase");}
    const float fast=recovery(3.3f,0,true),slow=recovery(4.5f,0,true),late=recovery(5.1f,0,true);
    check(fast>slow&&slow>late,"faster recovery must score higher");check(fast<=16.01f&&fast>12,"recovery reward remains bounded");close(late,0,"late recovery earns no bonus");
    for(int why=1;why<=4;++why)close(recovery(3.3f,why,true),0,"failed low airborne or truncated recovery cannot collect or later recover bonus");
    Pair reset;float before=recovery(3.3f,0,true,&reset),after=recovery(3.3f,0,true,&reset);close(before,after,"reset clears consumed phase and recovery state");
    // Ground-contact debounce ends the phase; sustained high lean still pays full penalty after landing.
    Pair landed;landed.step(frame(20,49,1.1f,true,25));
    for(int i=1;i<=5;++i)landed.step(frame(20+i*dt,51,1.1f,false,25));
    check(landed.step(frame(20+6*dt,51,1.1f,true,25))>0,"one contact frame is not confirmed landing");
    close(landed.step(frame(20+7*dt,51,1.1f,true,25)),0,"confirmed landing immediately ends exemption");
    std::cout<<"hurdle_phase_behavior_checks=passed\nfast_recovery_bonus="<<fast<<"\nslow_recovery_bonus="<<slow<<"\nnecessary_jump_relief="<<reliefTotal<<"\n";return 0;
}catch(const std::exception& e){std::cerr<<"hurdle_phase_behavior_checks=failed: "<<e.what()<<"\n";return 1;}}
