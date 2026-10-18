#include "reward_profiles.h"
#include <cmath>
#include <iostream>
#include <stdexcept>

void expect(float actual, float expected, const char* name) {
    if (!std::isfinite(actual) || std::abs(actual - expected) > 1e-4f)
        throw std::runtime_error(name);
}

int main() {
    const CourseSpec course = makeCourseSpec(GameMode::Classic100m);
    RewardState state;
    RgFrame frame{};
    frame.body_count = RG_PHYSICS_BODY_COUNT;
    frame.bodies[kTorsoBodySlot].y = -77.05f;
    frame.bodies[kHeadBodySlot].x = std::sin(25.0f * 3.14159265358979323846f / 180.0f) * 150;
    frame.bodies[kHeadBodySlot].y = -77.05f - std::cos(25.0f * 3.14159265358979323846f / 180.0f) * 150;
    frame.left_foot_grounded = 1;
    state.reset(course);
    expect(state.calculate(RewardProfile::Classic100m, course, frame, false), 0, "no instantaneous foot-contact bonus");
    frame.score = 0.1f;
    frame.sim_time = 1.0f / 30;
    expect(state.calculate(RewardProfile::Classic100m, course, frame, false), 0.24f + 0.2f / 30, "upright supported progress bonus");
    frame.left_foot_grounded = 0;
    frame.score = 0.2f;
    frame.sim_time += 1.0f / 30;
    expect(state.calculate(RewardProfile::Classic100m, course, frame, false), 0.12f + 0.2f / 30, "upright airborne progress is allowed");
    state.reset(course);
    frame.score = 0;
    frame.sim_time = 0;
    frame.bodies[kTorsoBodySlot].y = 150;
    const float lowRate = (429.8f - 150) / (429.8f + 77.05f) - 0.8f;
    expect(state.calculate(RewardProfile::Classic100m, course, frame, false), 0, "reset has no time penalty");
    float penalties = 0;
    for (int i = 1; i <= 12; ++i) {
        frame.sim_time = i / 30.0f;
        float r = state.calculate(RewardProfile::Classic100m, course, frame, false);
        if (i <= 5) expect(r, lowRate / 30, "brief low posture has only smooth height feedback");
        penalties += r;
    }
    expect(penalties, -0.04f + 0.4f * lowRate - 16.0f * 7 / 30, "time-based low-posture and pace penalties");
    frame.left_foot_grounded = 1;
    frame.score = 1.0f;
    frame.sim_time += 1.0f / 30;
    expect(state.calculate(RewardProfile::Classic100m, course, frame, false), (-16.2f + lowRate) / 30,
           "feet touching cannot exempt crawling or earn progress reward");
    frame.jump_landed = 1;
    frame.sim_time += 1.0f / 30;
    expect(state.calculate(RewardProfile::Classic100m, course, frame, false), -30 + (-16.2f + lowRate) / 30,
           "crawling finish cannot earn success bonus");
    frame.jump_landed = 0;
    frame.bodies[kTorsoBodySlot].y = -77.05f;
    frame.left_foot_grounded = 1;
    frame.sim_time += 1.0f / 30;
    expect(state.calculate(RewardProfile::Classic100m, course, frame, false), -15.8f / 30, "recovery cannot refund penalties or clear pace history");
    state.reset(course);
    frame.sim_time = 0;
    frame.score = 0.1f;
    expect(state.calculate(RewardProfile::Classic100m, course, frame, false), 0.24f, "episode reset clears progress and timers");
    state.reset(course);
    frame.score = 0;
    frame.failed = 1;
    expect(state.calculate(RewardProfile::Classic100m, course, frame, false), -30, "stronger failure penalty with richer speed rewards");
    frame.failed = 0;
    state.reset(course);
    expect(state.calculate(RewardProfile::Classic100m, course, frame, true), -30, "timeout penalized consistently with failure");
    state.reset(course);
    frame.jump_landed = 1;
    frame.sim_time = 30;
    expect(state.calculate(RewardProfile::Classic100m, course, frame, false), 176 - 16 * 30, "finish bonus remains additive to recorded idle time cost");
    state.reset(course);
    frame.jump_landed = 0;
    frame.sim_time = 1;
    frame.bodies[kTorsoBodySlot].y = -1000;
    expect(state.calculate(RewardProfile::Classic100m, course, frame, false), -15.8f, "height shaping remains capped and cannot offset standing pace deficit");
    std::cout << "foot_reward_behavior_checks=passed\n";
}
