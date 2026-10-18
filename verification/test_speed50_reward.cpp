#include "reward_profiles.h"
#include <cmath>
#include <iostream>
#include <stdexcept>

void expect(float actual, float expected, const char* name) {
    if (!std::isfinite(actual) || std::abs(actual - expected) > 2e-4f)
        throw std::runtime_error(name);
}

RgFrame pose(float lean = 7.5f) {
    RgFrame f{};
    f.body_count = RG_PHYSICS_BODY_COUNT;
    f.bodies[kTorsoBodySlot].y = -77.05f;
    const float a = lean * 3.14159265358979323846f / 180;
    f.bodies[kHeadBodySlot].x = std::sin(a) * 150;
    f.bodies[kHeadBodySlot].y = -77.05f - std::cos(a) * 150;
    f.left_foot_grounded = 1;
    return f;
}

void moveTorso(RgFrame& f, float worldX) {
    const float dx = worldX - f.bodies[kTorsoBodySlot].x;
    f.bodies[kTorsoBodySlot].x = worldX;
    f.bodies[kHeadBodySlot].x += dx;
}

void setProgress(RgFrame& f, float score) {
    f.score = score;
    moveTorso(f, score * kReferenceUnitsPerRaceMeter);
}

int main() {
    const auto course = makeCourseSpec(GameMode::Classic100m);
    RewardState s;
    for (float progress : {0.2f, 0.4f, 0.6f}) {
        s.reset(course);
        auto f = pose();
        s.calculate(RewardProfile::Classic100m, course, f, false);
        setProgress(f, progress);
        f.sim_time = 0.2f;
        const float quality = progress == 0.2f ? 0 : progress == 0.4f ? 5.0f / 7 : 1.0f;
        const float deficitCost = progress == 0.2f ? 3.2f * 1.44f / 4.84f : progress == 0.4f ? 3.2f * 0.04f / 4.84f : 0;
        expect(s.calculate(RewardProfile::Classic100m, course, f, false), progress * (2.8f + 6.0f * quality) + 0.04f - deficitCost, "speed reward focuses on 1.5 to 2.2m/s and remains capped");
        f.sim_time = 0.4f;
        expect(s.calculate(RewardProfile::Classic100m, course, f, false), -3.16f, "standing cannot farm speed reward or escape time cost");
        setProgress(f, f.score - 0.1f);
        f.sim_time = 0.6f;
        expect(s.calculate(RewardProfile::Classic100m, course, f, false), -3.16f, "backtracking cannot farm speed reward or escape time cost");
    }
    s.reset(course);
    auto f = pose(-15);
    s.calculate(RewardProfile::Classic100m, course, f, false);
    setProgress(f, 0.4f);
    f.sim_time = 0.2f;
    expect(s.calculate(RewardProfile::Classic100m, course, f, false), 0.96f + 0.04f - 0.00375f - 3.2f * 0.04f / 4.84f, "backward lean cannot earn speed bonus or escape pace cost");
    s.reset(course);
    f = pose();
    f.bodies[kTorsoBodySlot].y = 150;
    s.calculate(RewardProfile::Classic100m, course, f, false);
    setProgress(f, 0.4f);
    f.sim_time = 0.2f;
    const float lowRate = (429.8f - 150) / (429.8f + 77.05f) - 0.8f;
    expect(s.calculate(RewardProfile::Classic100m, course, f, false), 0.2f * lowRate - 3.2f * 0.04f / 4.84f, "low posture cannot earn speed reward or escape pace cost");
    s.reset(course);
    f = pose();
    f.score = 0.1f;
    f.sim_time = 1.0f / 30;
    expect(s.calculate(RewardProfile::Classic100m, course, f, false), 0.28f + 0.2f / 30, "reset clears pace state");
    s.reset(course);
    f = pose();
    s.calculate(RewardProfile::Classic100m, course, f, false);
    f.score = 0.4f;
    moveTorso(f, 0.44f * kReferenceUnitsPerRaceMeter);
    f.sim_time = 0.2f;
    expect(s.calculate(RewardProfile::Classic100m, course, f, false), 3.56f,
           "precise torso pace reaches target despite score quantization");
    f.score = 0.3f;
    moveTorso(f, 0.34f * kReferenceUnitsPerRaceMeter);
    f.sim_time = 0.4f;
    expect(s.calculate(RewardProfile::Classic100m, course, f, false), -3.16f, "body backtracking cannot earn pace");
    f.score = 0.4f;
    moveTorso(f, 0.44f * kReferenceUnitsPerRaceMeter);
    f.sim_time = 0.6f;
    expect(s.calculate(RewardProfile::Classic100m, course, f, false), -3.16f, "returning to previous body maximum cannot earn pace");
    for (float t : {65.0f, 50.0f, 48.0f, 30.0f}) {
        s.reset(course);
        f = pose();
        f.jump_landed = 1;
        f.sim_time = t;
        const float bonus = t >= 65 ? 0 : t <= 48 ? 80 : 80.0f * 15 / 17;
        expect(s.calculate(RewardProfile::Classic100m, course, f, false), 120 - t + 0.2f * t + bonus - 16 * t, "fast finish bonus remains additive to recorded idle cost");
        f.failed = 1;
        s.reset(course);
        expect(s.calculate(RewardProfile::Classic100m, course, f, false), -30 + 0.2f * t - 16 * t, "failure cannot collect fast finish bonus");
    }
    std::cout << "speed50_reward_behavior_checks=passed\n";
}
