#include "reward_profiles.h"

#include <algorithm>
#include <cmath>

namespace {
constexpr float kTrackY = 429.8f;
constexpr float kHurdleRewardHeight = 381.5f;
constexpr float kStandingBodyY = -77.05f;
constexpr float kLeadFootHeightRewardCap = 8.0f;
constexpr float kLeadFootHeightCapScale = 1.1f;
constexpr float kRearFootHeightRewardCap = 8.0f;
constexpr float kBodyHeightRewardCap = 5.0f;
constexpr float kBodyHeightTargetHurdleScale = 0.45f;
constexpr float kLongJumpLandingDistanceScale = 5.0f;
constexpr float kLongJumpPostLandingProgressScale = 0.5f;
// Foot-running reward v12. Knee contacts are not exposed by the backend;
// sustained low torso height is a posture proxy, including feet-on-ground crawl.
constexpr float kUprightHeightRatio = 0.8f;
constexpr float kSupportedProgressBonus = 1.0f;
constexpr float kLowPostureGraceSeconds = 0.2f;
constexpr float kLowPosturePenaltyPerSecond = 0.2f;
constexpr float kMaximumLowPostureFinishFraction = 0.1f;
constexpr float kPostureRewardPerSecond = 1.0f;
constexpr float kProgressRewardScale = 1.2f;
constexpr float kLeanTargetMinDegrees = 5.0f;
constexpr float kLeanTargetMaxDegrees = 10.0f;
constexpr float kLeanToleranceDegrees = 15.0f;
constexpr float kForwardLeanProgressBonus = 0.4f;
constexpr float kBackwardLeanLimitDegrees = -10.0f;
constexpr float kBackwardLeanGraceSeconds = 0.15f;
constexpr float kBackwardLeanPenaltyPerSecond = 0.3f;
constexpr float kPaceWindowSeconds = 0.2f;
constexpr float kTargetProgressSpeed = 2.2f;
constexpr float kSpeedRewardMinimum = 1.5f;
constexpr float kSpeedProgressBonus = 6.0f;
constexpr float kPaceDeficitPenaltyPerSecond = 16.0f;
constexpr float kClassicFailurePenalty = 30.0f;
constexpr float kFastFinishReferenceSeconds = 65.0f;
constexpr float kFastFinishTargetSeconds = 48.0f;
constexpr float kFastFinishBonus = 80.0f;
constexpr float kStartMilestoneMeters = 10.0f;
constexpr float kStartReferenceSeconds = 7.5f;
constexpr float kStartTargetSeconds = 5.5f;
constexpr float kFastStartBonus = 40.0f;
constexpr float kStartProgressBonusPerMeter = 4.0f;
constexpr float kStartProgressDeadlineSeconds = 9.0f;

float forwardLeanDegrees(const RgFrame& frame) {
    // Calibrate from world geometry: zero is head above torso, positive is
    // towards the finish (+x). The raw torso angle has a different zero.
    const auto& head = frame.bodies[kHeadBodySlot];
    const auto& torso = frame.bodies[kTorsoBodySlot];
    return std::atan2(head.x - torso.x, torso.y - head.y) * 180.0f / 3.14159265358979323846f;
}

float scoreFromReferenceX(float x) {
    return std::round(x / kReferenceUnitsPerPhysicsUnit) / 10.0f;
}

float heightRewardRatio(float y) {
    const float height = std::max(0.0f, kTrackY - y);
    return std::clamp(height / (kHurdleRewardHeight * kLeadFootHeightCapScale), 0.0f, 1.0f);
}

float bodyHeightRewardRatio(float bodyY) {
    const float targetLift = kHurdleRewardHeight * kBodyHeightTargetHurdleScale;
    const float bodyLift = std::max(0.0f, kStandingBodyY - bodyY);
    return std::clamp(bodyLift / targetLift, 0.0f, 1.0f);
}
}

void RewardState::reset(const CourseSpec& course) {
    hurdles_.assign(course.hurdleLocations.size(), HurdleRewardState{});
    progressScore_ = 0.0f;
    previousRewardTime_ = 0.0f;
    lowPostureTime_ = 0.0f;
    penalizedLowPostureSeconds_ = 0.0f;
    backwardLeanTime_ = 0.0f;
    paceWindowTime_ = 0.0f;
    paceWindowProgress_ = 0.0f;
    recentProgressSpeed_ = 0.0f;
    paceMeasurementReady_ = false;
    paceAnchorReady_ = false;
    maximumTorsoX_ = 0.0f;
    startMilestoneSeen_ = false;
    longJumpLandingSeen_ = false;
    longJumpPostLandingX_ = 0.0f;
}

float RewardState::calculate(
    RewardProfile profile,
    const CourseSpec& course,
    const RgFrame& frame,
    bool truncated) {
    switch (profile) {
    case RewardProfile::Classic100m: {
        const float cappedProgress = std::clamp(frame.score, 0.0f, 100.0f);
        const float startProgress = std::max(0.0f,
            std::min(cappedProgress, kStartMilestoneMeters)
                - std::min(progressScore_, kStartMilestoneMeters));
        const float progressReward = std::max(0.0f, cappedProgress - progressScore_);
        float reward = 0.0f;
        progressScore_ = std::max(progressScore_, cappedProgress);
        const float elapsed = std::max(0.0f, frame.sim_time - previousRewardTime_);
        previousRewardTime_ = frame.sim_time;
        // World torso displacement is more precise than the score's 0.1m steps.
        // Keep a maximum so backward/forward oscillation cannot inflate pace.
        const float torsoX = frame.bodies[kTorsoBodySlot].x;
        float torsoProgressMeters = 0.0f;
        if (!paceAnchorReady_) {
            maximumTorsoX_ = torsoX;
            paceAnchorReady_ = true;
        } else {
            torsoProgressMeters = std::max(0.0f, torsoX - maximumTorsoX_) / kReferenceUnitsPerRaceMeter;
            maximumTorsoX_ = std::max(maximumTorsoX_, torsoX);
        }
        if (elapsed > 0.0f) {
            paceWindowTime_ += elapsed;
            paceWindowProgress_ += torsoProgressMeters;
            if (paceWindowTime_ + 1e-6f >= kPaceWindowSeconds) {
                recentProgressSpeed_ = paceWindowProgress_ / paceWindowTime_;
                paceMeasurementReady_ = true;
                paceWindowTime_ = 0.0f;
                paceWindowProgress_ = 0.0f;
            }
        }
        const float standingHeight = kTrackY - kStandingBodyY;
        const float torsoHeight = kTrackY - frame.bodies[kTorsoBodySlot].y;
        const float heightRatio = std::clamp(torsoHeight / standingHeight, 0.0f, 1.0f);
        const bool upright = torsoHeight >= kUprightHeightRatio * standingHeight;
        const bool footSupported = frame.left_foot_grounded || frame.right_foot_grounded;
        const float leanDegrees = forwardLeanDegrees(frame);
        if (upright) {
            reward = kProgressRewardScale * progressReward;
        }
        if (upright && footSupported) {
            reward += kProgressRewardScale * kSupportedProgressBonus * progressReward;
            const float targetError = std::max({kLeanTargetMinDegrees - leanDegrees,
                leanDegrees - kLeanTargetMaxDegrees, 0.0f});
            const float leanQuality = std::clamp(1.0f - targetError / kLeanToleranceDegrees, 0.0f, 1.0f);
            reward += kForwardLeanProgressBonus * progressReward * leanQuality;
        }
        if (upright && leanDegrees >= kBackwardLeanLimitDegrees && leanDegrees <= 30.0f) {
            const float speedQuality = std::clamp(
                (recentProgressSpeed_ - kSpeedRewardMinimum) / (kTargetProgressSpeed - kSpeedRewardMinimum), 0.0f, 1.0f);
            reward += kSpeedProgressBonus * progressReward * speedQuality;
            // Give acceleration feedback before crossing 10m. Reward only new
            // distance, taper with game time, and cap the total at 40 points.
            if (!frame.failed && !truncated) {
                const float startProgressQuality = std::clamp(
                    (kStartProgressDeadlineSeconds - frame.sim_time)
                        / (kStartProgressDeadlineSeconds - kStartTargetSeconds), 0.0f, 1.0f);
                reward += kStartProgressBonusPerMeter * startProgress * startProgressQuality;
            }
        }
        if (paceMeasurementReady_) {
            const float speedDeficit = std::clamp(1.0f - recentProgressSpeed_ / kTargetProgressSpeed, 0.0f, 1.0f);
            reward -= kPaceDeficitPenaltyPerSecond * elapsed * speedDeficit * speedDeficit;
        }
        const float previousBackwardTime = backwardLeanTime_;
        const bool backwardLeaning = upright && leanDegrees < kBackwardLeanLimitDegrees;
        backwardLeanTime_ = backwardLeaning ? backwardLeanTime_ + elapsed : 0.0f;
        if (backwardLeaning) {
            const float penalizedBackwardTime = std::max(0.0f, backwardLeanTime_ - kBackwardLeanGraceSeconds)
                - std::max(0.0f, previousBackwardTime - kBackwardLeanGraceSeconds);
            const float severity = std::clamp((kBackwardLeanLimitDegrees - leanDegrees) / 20.0f, 0.0f, 1.0f);
            reward -= kBackwardLeanPenaltyPerSecond * penalizedBackwardTime * severity;
        }
        // Dense height feedback helps discover balance before forward progress.
        // Capped at the initial standing height to avoid rewarding excessive jumps.
        reward += kPostureRewardPerSecond * elapsed * (heightRatio - kUprightHeightRatio);
        const float previousLowTime = lowPostureTime_;
        lowPostureTime_ = !upright ? lowPostureTime_ + elapsed : 0.0f;
        const float penalizedTime = std::max(0.0f, lowPostureTime_ - kLowPostureGraceSeconds)
            - std::max(0.0f, previousLowTime - kLowPostureGraceSeconds);
        if (!upright) {
            reward -= kLowPosturePenaltyPerSecond * penalizedTime;
            penalizedLowPostureSeconds_ += penalizedTime;
        }
        // A nearby timed milestone gives the initial acceleration direct credit.
        // Consume it on the first crossing, including unqualified posture.
        if (!startMilestoneSeen_ && cappedProgress >= kStartMilestoneMeters) {
            startMilestoneSeen_ = true;
            if (upright && !frame.failed && !truncated
                && leanDegrees >= kBackwardLeanLimitDegrees && leanDegrees <= 30.0f
                && penalizedLowPostureSeconds_ <= kMaximumLowPostureFinishFraction * frame.sim_time) {
                const float startQuality = std::clamp(
                    (kStartReferenceSeconds - frame.sim_time) / (kStartReferenceSeconds - kStartTargetSeconds), 0.0f, 1.0f);
                reward += kFastStartBonus * startQuality;
            }
        }
        if (frame.failed || truncated) {
            reward -= kClassicFailurePenalty;
        } else if (frame.jump_landed) {
            if (penalizedLowPostureSeconds_ <= kMaximumLowPostureFinishFraction * frame.sim_time) {
                reward += 120.0f - frame.sim_time;
                const float finishSpeedQuality = std::clamp(
                    (kFastFinishReferenceSeconds - frame.sim_time)
                        / (kFastFinishReferenceSeconds - kFastFinishTargetSeconds), 0.0f, 1.0f);
                reward += kFastFinishBonus * finishSpeedQuality;
            } else {
                reward -= kClassicFailurePenalty;
            }
        }
        return reward;
    }
    case RewardProfile::Hurdle110m: {
        const float cappedProgress = std::clamp(frame.score, 0.0f, 110.0f);
        float reward = std::max(0.0f, cappedProgress - progressScore_);
        progressScore_ = std::max(progressScore_, cappedProgress);

        if (!frame.failed) {
            const size_t activeCount = std::min<size_t>(frame.active_hurdle_count, hurdles_.size());
            for (size_t i = 0; i < activeCount; ++i) {
                HurdleRewardState& hurdle = hurdles_[i];
                const float location = course.hurdleLocations[i];
                const RgBodySample& leadFoot = frame.bodies[kLeftFootBodySlot];
                const RgBodySample& rearFoot = frame.bodies[kRightFootBodySlot];
                const RgBodySample& body = frame.bodies[kTorsoBodySlot];
                if (!hurdle.leadFootApplied && leadFoot.x >= location) {
                    hurdle.leadFootApplied = true;
                    reward += heightRewardRatio(leadFoot.y) * kLeadFootHeightRewardCap;
                }
                if (!hurdle.bodyApplied && body.x >= location) {
                    hurdle.bodyApplied = true;
                    reward += bodyHeightRewardRatio(body.y) * kBodyHeightRewardCap;
                }
                if (!hurdle.rearFootApplied && rearFoot.x >= location) {
                    hurdle.rearFootApplied = true;
                    reward += heightRewardRatio(rearFoot.y) * kRearFootHeightRewardCap;
                }
            }
        }
        if (frame.failed) {
            reward -= 10.0f;
        } else if (frame.race_finished) {
            reward += 120.0f - frame.sim_time;
        }
        return reward;
    }
    case RewardProfile::LongJump30m: {
        const float runupProgress = std::clamp(
            scoreFromReferenceX(frame.view_target_x),
            0.0f,
            kLongJumpTakeoffDistance);
        float reward = std::max(0.0f, runupProgress - progressScore_);
        progressScore_ = std::max(progressScore_, runupProgress);

        if (frame.long_jump_landing_started) {
            if (!longJumpLandingSeen_) {
                longJumpLandingSeen_ = true;
                longJumpPostLandingX_ = frame.long_jump_landing_body_x;
            }
            const float currentX = std::min(frame.bodies[kTorsoBodySlot].x, course.sandPitEndLocation);
            const float walkingDistance = std::max(
                0.0f,
                (currentX - longJumpPostLandingX_) / kReferenceUnitsPerRaceMeter);
            reward += walkingDistance * kLongJumpPostLandingProgressScale;
            longJumpPostLandingX_ = std::max(longJumpPostLandingX_, currentX);
        }
        if (frame.long_jump_exited_sand) {
            reward += frame.long_jump_distance * kLongJumpLandingDistanceScale;
        }
        if (frame.failed) {
            reward -= 10.0f;
        }
        return reward;
    }
    }
    return 0.0f;
}
