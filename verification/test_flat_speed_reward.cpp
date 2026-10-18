#include "reward_profiles.h"
#include "flat_speed_baseline_reward.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
namespace {
void close(float a,float b,const char* label,float tolerance=.003f){if(!std::isfinite(a)||!std::isfinite(b)||std::abs(a-b)>tolerance)throw std::runtime_error(std::string(label)+": "+std::to_string(a)+" vs "+std::to_string(b));}
void check(bool pass,const char* label){if(!pass)throw std::runtime_error(label);}
RgFrame pose(float t,float meters,float height=1,float lean=5){RgFrame f{};f.body_count=12;f.sim_time=t;f.score=meters;f.left_foot_grounded=f.right_foot_grounded=1;
f.bodies[kTorsoBodySlot].x=meters*400;f.bodies[kTorsoBodySlot].y=429.8f-height*506.85f;
f.bodies[kHeadBodySlot].x=meters*400+150*std::sin(lean*3.141592653589793f/180);f.bodies[kHeadBodySlot].y=f.bodies[kTorsoBodySlot].y-150*std::cos(lean*3.141592653589793f/180);return f;}
struct Pair{CourseSpec course=makeCourseSpec(GameMode::Classic100m);RewardState current;SpeedBaselineRewardState old;Pair(){course.hurdleLocations={500*400};current.reset(course);old.reset(course);}float step(const RgFrame& f,bool truncated=false){return current.calculate(RewardProfile::Classic100m,course,f,truncated)-old.calculate(RewardProfile::Classic100m,course,f,truncated);}};
struct Measured{float difference,pace,progress,newCoefficient;};
Measured sample(float speed,float endpoint,float height=1,float lean=5){Pair p;auto first=pose(20,endpoint-.2f*speed,height,lean);p.step(first);auto last=pose(20.2f,endpoint,height,lean);
float difference=p.step(last);float pace=std::max(0.0f,last.bodies[kTorsoBodySlot].x-first.bodies[kTorsoBodySlot].x)/400/(last.sim_time-first.sim_time);
float progress=std::max(0.0f,std::clamp(last.score,0.0f,100.0f)-std::clamp(first.score,0.0f,100.0f));
float oldCoefficient=6*std::clamp((pace-1.5f)/(2.5f-1.5f),0.0f,1.0f);
float coefficient=progress>0?difference/progress+oldCoefficient:oldCoefficient;return{difference,pace,progress,coefficient};}
}
int main(){try{
    for(float endpoint:{1.0f,9.99f,45.0f,50.0f,60.0f,64.99f})for(float speed:{0.0f,1.5f,2.4f,2.6f,2.8f,3.0f})close(sample(speed,endpoint).difference,0,"non-flat reward remains exactly baseline");
    for(float endpoint:{10.0f,20.0f,44.99f,65.0f,90.0f,100.0f}){
        float previous=-1;
        for(float speed:{0.5f,1.5f,2.0f,2.4f,2.5f,2.6f,2.8f,3.0f,5.0f}){
            auto m=sample(speed,endpoint);
            float newCoefficient=8*std::clamp((m.pace-1.5f)/(2.8f-1.5f),0.0f,1.0f);
            float oldCoefficient=6*std::clamp((m.pace-1.5f)/(2.5f-1.5f),0.0f,1.0f);
            close(m.difference,m.progress*(newCoefficient-oldCoefficient),"only approved local speed-progress term changes");
            close(m.newCoefficient,newCoefficient,"new progress coefficient matches independent formula");
            check(m.newCoefficient>=previous-.003f,"flat speed coefficient is monotonic");
            check(m.newCoefficient<=8.003f&&m.newCoefficient>=-.003f,"flat coefficient remains bounded zero to eight");previous=m.newCoefficient;
        }
    }
    close(sample(-2.4f,25).difference,0,"backtracking cannot collect new speed-progress reward");
    close(sample(0,25).difference,0,"no progress cannot collect new speed-progress reward");
    close(sample(2.8f,25,.6f).difference,0,"low posture still disables speed-progress reward");
    close(sample(2.8f,25,1,-20).difference,0,"severe backward lean still disables speed-progress reward");
    // Reaching the capped finish can receive the last remaining progress once;
    // any further score growth or backwards/forwards oscillation cannot repeat.
    Pair finish;finish.step(pose(20,99));finish.step(pose(20.4f,100));
    close(finish.step(pose(20.6f,101)),0,"beyond capped finish has no additional speed-progress reward");
    close(finish.step(pose(20.8f,99)),0,"finish backtracking has no new reward");
    close(finish.step(pose(21,101)),0,"re-crossing capped finish cannot repeat reward");
    std::cout<<"flat_speed_reward_behavior_checks=passed\ncoefficient_at_2p4="<<sample(2.4f,25).newCoefficient<<"\ncoefficient_at_2p6="<<sample(2.6f,25).newCoefficient<<"\ncoefficient_at_2p8="<<sample(2.8f,25).newCoefficient<<"\n";return 0;
}catch(const std::exception& e){std::cerr<<"flat_speed_reward_behavior_checks=failed: "<<e.what()<<"\n";return 1;}}
