#include "reward_profiles.h"
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

constexpr float kTrack = 429.8f;
constexpr float kHeight = 506.85f;

void expect(float actual, float expected, const char* name) {
    if (!std::isfinite(actual) || std::abs(actual - expected) > 1e-3f)
        throw std::runtime_error(std::string(name) + ": actual=" + std::to_string(actual) + ", expected=" + std::to_string(expected));
}

RgFrame frame(float time = 0, float score = 0, float height = 1, float lean = 0) {
    RgFrame f{};
    f.body_count = RG_PHYSICS_BODY_COUNT;
    f.sim_time = time;
    f.score = score;
    f.bodies[kTorsoBodySlot].x = score * 400;
    f.bodies[kTorsoBodySlot].y = kTrack - height * kHeight;
    const float radians = lean * 3.14159265358979323846f / 180;
    f.bodies[kHeadBodySlot].x = f.bodies[kTorsoBodySlot].x + 150 * std::sin(radians);
    f.bodies[kHeadBodySlot].y = f.bodies[kTorsoBodySlot].y - 150 * std::cos(radians);
    f.left_foot_grounded = 1;
    return f;
}

float reward(RewardState& state, const CourseSpec& course, const RgFrame& f, bool truncated = false) {
    return state.calculate(RewardProfile::Classic100mTask, course, f, truncated);
}

double successfulEpisode(const CourseSpec& course, int duration) {
    RewardState state;
    state.reset(course);
    reward(state, course, frame());
    double total = 0;
    const int steps = duration * 30;
    for (int i = 1; i <= steps; ++i) {
        auto f = frame(i / 30.0f, 100.0f * i / steps);
        f.jump_landed = i == steps;
        total += reward(state, course, f);
    }
    return total;
}

int main() {
    try {
        const auto course = makeCourseSpec(GameMode::Classic100m);
        if (inferGameMode(RewardProfile::Classic100mTask) != GameMode::Classic100m
            || course.finishType != FinishType::JumpLanding || course.timeLimit != 120
            || course.hurdleLocations.size() != 1 || course.hurdleLocations[0] != 50 * 400)
            throw std::runtime_error("Task profile must retain classic rules");
        RewardState plain, changed;
        plain.reset(course); changed.reset(course);
        reward(plain, course, frame()); reward(changed, course, frame());
        float totalPlain = 0, totalChanged = 0;
        for (int i = 1; i <= 300; ++i) {
            auto a = frame(i / 30.0f, i / 12.0f, .95f, 5);
            auto b = frame(i / 30.0f, i / 12.0f, .81f + .30f * (1 + std::sin(i * .31f)) / 2,
                           -40 + 90 * (1 + std::sin(i * .73f)) / 2);
            b.left_foot_grounded = i % 19 < 5;
            b.right_foot_grounded = i % 13 < 4;
            const float ra = reward(plain, course, a);
            const float rb = reward(changed, course, b);
            expect(ra, rb, "Angles, upright vertical excursion and contacts must not shape task reward");
            totalPlain += ra; totalChanged += rb;
        }
        expect(totalPlain, totalChanged, "Matched task trajectories must have identical reward");
        const double at43 = successfulEpisode(course, 43);
        const double at45 = successfulEpisode(course, 45);
        expect(at43, 3000 - 21 * 43, "43s successful task return");
        expect(at45, 3000 - 21 * 45, "45s successful task return");
        expect(at43 - at45, 42, "Saving two seconds must improve return by 42");

        for (bool truncated : {false, true}) {
            RewardState state; state.reset(course); reward(state, course, frame());
            auto f = frame(43, 100); f.jump_landed = 1; f.failed = !truncated;
            expect(reward(state, course, f, truncated), 200 - 43 - 200,
                   "Failed or truncated endpoint receives no success bonus");
        }
        RewardState grace; grace.reset(course); reward(grace, course, frame());
        expect(reward(grace, course, frame(.1f, 0, .7f)), -.1f, "First .1s low posture is within grace");
        expect(reward(grace, course, frame(.2f, 0, .7f)), -.1f, "First .2s low posture is within grace");
        expect(reward(grace, course, frame(.3f, 0, .7f)), -.6f, "Low posture beyond grace is penalized");
        expect(reward(grace, course, frame(.4f, 0, 1)), -.1f, "Standing cannot refund low posture penalties");

        RewardState reject; reject.reset(course); reward(reject, course, frame());
        expect(reward(reject, course, frame(5, 0, .7f)), -29, "Accumulate 4.8 seconds of penalized low posture");
        expect(reward(reject, course, frame(5.1f, 0, 1)), -.1f, "Low-to-upright keeps historical total");
        auto rejectedFinish = frame(10, 100); rejectedFinish.jump_landed = 1;
        expect(reward(reject, course, rejectedFinish), -4.9f, "Low fraction cannot be erased by standing before finish");
        RewardState allowed; allowed.reset(course); reward(allowed, course, frame());
        reward(allowed, course, frame(1, 0, .7f)); reward(allowed, course, frame(1.1f));
        auto allowedFinish = frame(10, 100); allowedFinish.jump_landed = 1;
        expect(reward(allowed, course, allowedFinish), 200 - 8.9f + 2600, "Low fraction below .1 remains qualified");

        RewardState progress; progress.reset(course); reward(progress, course, frame());
        expect(reward(progress, course, frame(1, 5)), 9, "First new max progress is rewarded");
        expect(reward(progress, course, frame(2, 4)), -1, "Backward score gains no progress reward");
        expect(reward(progress, course, frame(3, 5)), -1, "Recovering old max gains no progress reward");
        expect(reward(progress, course, frame(4, 6)), 1, "Only new max gains reward");
        progress.reset(course); reward(progress, course, frame());
        expect(reward(progress, course, frame(1, 5)), 9, "Reset clears previous max and low history");
        RewardState lowProgress; lowProgress.reset(course); reward(lowProgress, course, frame());
        reward(lowProgress, course, frame(1, 5, .7f));
        expect(reward(lowProgress, course, frame(2, 5)), -1, "Standing cannot retroactively collect crawl progress");

        RewardState endpoint; endpoint.reset(course); reward(endpoint, course, frame());
        auto scoreOnly = frame(43, 100); scoreOnly.race_finished = 1; scoreOnly.jumped = 1;
        expect(reward(endpoint, course, scoreOnly), 157, "Score or race_finished alone cannot receive jump_landed bonus");
        auto landing = scoreOnly; landing.sim_time += 1.0f / 30; landing.jump_landed = 1;
        expect(reward(endpoint, course, landing), -(landing.sim_time - 43) + 2800 - 20 * landing.sim_time,
               "Original jump_landed endpoint receives success bonus");
        std::cout << "task_only_reward_behavior_checks=passed; 43s=" << at43 << "; 45s=" << at45
                  << "; difference=" << at43 - at45 << "\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "task_only_reward_behavior_checks=failed: " << error.what() << "\n";
        return 1;
    }
}
