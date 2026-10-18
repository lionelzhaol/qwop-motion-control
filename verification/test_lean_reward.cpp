#include "reward_profiles.h"
#include <cmath>
#include <iostream>
#include <stdexcept>

void expect(float actual, float expected, const char* name) {
    if (!std::isfinite(actual) || std::abs(actual - expected) > 1e-4f)
        throw std::runtime_error(name);
}

RgFrame pose(float lean) {
    RgFrame f{};
    f.body_count = RG_PHYSICS_BODY_COUNT;
    f.bodies[kTorsoBodySlot].y = -77.05f;
    const float a = lean * 3.14159265358979323846f / 180.0f;
    f.bodies[kHeadBodySlot].x = std::sin(a) * 150;
    f.bodies[kHeadBodySlot].y = -77.05f - std::cos(a) * 150;
    f.left_foot_grounded = 1;
    return f;
}

int main() {
    const auto course = makeCourseSpec(GameMode::Classic100m);
    RewardState s;
    for (float angle : {5.0f, 7.5f, 10.0f}) {
        s.reset(course);
        auto f = pose(angle);
        expect(s.calculate(RewardProfile::Classic100m, course, f, false), 0, "reset has no lean reward");
        f.score = 0.1f;
        f.sim_time = 1.0f / 30;
        expect(s.calculate(RewardProfile::Classic100m, course, f, false), 0.28f + 0.2f / 30, "forward supported progress earns lean bonus");
        f.sim_time += 1.0f / 30;
        expect(s.calculate(RewardProfile::Classic100m, course, f, false), 0.2f / 30, "leaning without new progress cannot earn lean bonus");
        f.score = 0.05f;
        f.sim_time += 1.0f / 30;
        expect(s.calculate(RewardProfile::Classic100m, course, f, false), 0.2f / 30, "walking backward cannot repeat progress bonus");
    }
    s.reset(course);
    auto f = pose(7.5f);
    f.left_foot_grounded = 0;
    f.score = 0.1f;
    f.sim_time = 1.0f / 30;
    expect(s.calculate(RewardProfile::Classic100m, course, f, false), 0.12f + 0.2f / 30, "airborne progress allowed without foot lean bonus");
    for (float angle : {-30.0f, -15.0f}) {
        s.reset(course);
        f = pose(angle);
        float total = 0;
        for (int i = 1; i <= 12; ++i) {
            f.sim_time = i / 30.0f;
            const float r = s.calculate(RewardProfile::Classic100m, course, f, false);
            if (i <= 4) expect(r, 0.2f / 30, "brief backward sway grace");
            total += r;
        }
        expect(total, (angle == -30.0f ? 0.005f : 0.06125f) - 16 * 7.0f / 30, "backward penalty remains graded alongside pace deficit");
        f = pose(7.5f);
        f.sim_time = 13.0f / 30;
        expect(s.calculate(RewardProfile::Classic100m, course, f, false), -15.8f / 30, "posture recovery cannot refund penalties");
        s.reset(course);
        f = pose(-30);
        f.sim_time = 1.0f / 30;
        expect(s.calculate(RewardProfile::Classic100m, course, f, false), 0.2f / 30, "reset clears backward timer");
    }
    s.reset(course);
    f = pose(80);
    f.score = 0.1f;
    expect(s.calculate(RewardProfile::Classic100m, course, f, false), 0.24f, "excessive forward lean does not earn target bonus");
    std::cout << "lean_reward_behavior_checks=passed\n";
}
