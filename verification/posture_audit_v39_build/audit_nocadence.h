#pragma once

#include "course_rules.h"
#include "physics_backend_api.h"

#include <vector>

class AuditNoCadenceRewardState {
public:
    void reset(const CourseSpec& course);
    float calculate(RewardProfile profile, const CourseSpec& course, const RgFrame& frame, bool truncated);

private:
    struct HurdleRewardState {
        bool leadFootApplied = false;
        bool bodyApplied = false;
        bool rearFootApplied = false;
    };

    std::vector<HurdleRewardState> hurdles_;
    float progressScore_ = 0.0f;
    float previousRewardTime_ = 0.0f;
    float lowPostureTime_ = 0.0f;
    float penalizedLowPostureSeconds_ = 0.0f;
    float backwardLeanTime_ = 0.0f;
    float paceWindowTime_ = 0.0f;
    float paceWindowProgress_ = 0.0f;
    float recentProgressSpeed_ = 0.0f;
    bool paceMeasurementReady_ = false;
    bool paceAnchorReady_ = false;
    float maximumTorsoX_ = 0.0f;
    bool startMilestoneSeen_ = false;
    bool earlyStartMilestoneSeen_ = false;
    float smoothedTorsoY_ = 0.0f;
    bool torsoHeightAnchorReady_ = false;
    float footAirSeconds_[2] = {0.0f, 0.0f};
    float footGroundSeconds_[2] = {0.0f, 0.0f};
    bool footLiftConfirmed_[2] = {false, false};
    int lastLandingFoot_ = -1;
    float lastLandingTime_ = 0.0f;
    float lastLandingProgress_ = 0.0f;
    bool classicHurdleArmed_ = false;
    bool classicHurdleCrossed_ = false;
    bool classicHurdleJumpSeen_ = false;
    bool classicHurdleJumpClosed_ = false;
    bool classicHurdleLandingConfirmed_ = false;
    bool classicHurdleRecoverySeen_ = false;
    float classicHurdleCrossTime_ = 0.0f;
    float classicHurdleJumpTime_ = 0.0f;
    float classicHurdleAirTime_ = 0.0f;
    float classicHurdleGroundTime_ = 0.0f;
    bool longJumpLandingSeen_ = false;
    float longJumpPostLandingX_ = 0.0f;
};

