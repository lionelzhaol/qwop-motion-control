#include "audit_nocadence.h"

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
// Foot-running reward v39. Knee contacts are not exposed by the backend;
// sustained low torso height is a posture proxy, including feet-on-ground crawl.
constexpr float kUprightHeightRatio = 0.8f;
constexpr float kSupportedProgressBonus = 1.0f;
constexpr float kLowPostureGraceSeconds = 0.2f;
constexpr float kLowPosturePenaltyPerSecond = 0.2f;
constexpr float kMaximumLowPostureFinishFraction = 0.1f;
constexpr float kPostureRewardPerSecond = 1.0f;
constexpr float kProgressRewardScale = 1.2f;
constexpr float kLeanTargetMinDegrees = 5.0f;
constexpr float kLeanTargetMaxDegrees = 5.0f;
constexpr float kLeanToleranceDegrees = 10.0f;
constexpr float kForwardLeanProgressBonus = 1.0f;
constexpr float kLeanPostureGraceDegrees = 3.0f;
constexpr float kLeanPostureToleranceDegrees = 12.0f;
constexpr float kLeanPosturePenaltyPerSecond = 2.0f;
constexpr float kBackwardLeanLimitDegrees = -10.0f;
constexpr float kBackwardLeanGraceSeconds = 0.15f;
constexpr float kBackwardLeanPenaltyPerSecond = 0.3f;
constexpr float kPaceWindowSeconds = 0.2f;
constexpr float kTargetProgressSpeed = 2.5f;
constexpr float kSpeedRewardMinimum = 1.5f;
constexpr float kSpeedProgressBonus = 6.0f;
constexpr float kFlatTargetProgressSpeed = 2.8f;
constexpr float kFlatSpeedProgressBonus = 8.0f;
constexpr float kPaceDeficitPenaltyPerSecond = 16.0f;
constexpr float kClassicFailurePenalty = 30.0f;
constexpr float kFastFinishReferenceSeconds = 65.0f;
constexpr float kFastFinishTargetSeconds = 43.0f;
constexpr float kFastFinishBonus = 1200.0f;
constexpr float kStartMilestoneMeters = 10.0f;
constexpr float kStartReferenceSeconds = 7.5f;
constexpr float kStartTargetSeconds = 5.5f;
constexpr float kFastStartBonus = 40.0f;
constexpr float kStartProgressBonusPerMeter = 4.0f;
constexpr float kStartProgressDeadlineSeconds = 9.0f;
constexpr float kEarlyStartMeters = 3.0f;
constexpr float kEarlyStartTargetSeconds = 2.0f;
constexpr float kEarlyStartDeadlineSeconds = 4.0f;
constexpr float kEarlyStartBonus = 12.0f;
constexpr float kEarlyStartProgressBonusPerMeter = 2.0f;
constexpr float kHeightSmoothingSeconds = 0.6f;
constexpr float kHeightOscillationGrace = 0.02f;
constexpr float kHeightOscillationTolerance = 0.06f;
constexpr float kVerticalMotionPenaltyPerSecond = 4.0f;
constexpr float kFootAirConfirmationSeconds = 0.1f;
constexpr float kFootGroundConfirmationSeconds = 2.0f / 30.0f;
constexpr float kMinimumLandingIntervalSeconds = 0.15f;
constexpr float kCadenceBaseline = 1.0f;
constexpr float kCadenceTarget = 1.4f;
constexpr float kCadenceProgressBonus = 0.0f;
constexpr float kHurdleArmMeters = 2.0f;
constexpr float kHurdleJumpBeforeMeters = 1.0f;
constexpr float kHurdleJumpAfterMeters = 3.5f;
constexpr float kHurdleJumpHeightRatio = 1.03f;
constexpr float kHurdleMaximumJumpSeconds = 1.8f;
constexpr float kHurdleLandingAfterMeters = 0.25f;
constexpr float kHurdlePhaseEndMeters = 4.0f;
constexpr float kHurdleRecoveryMeters = 8.0f;
constexpr float kHurdleRecoveryTargetSeconds = 3.3f;
constexpr float kHurdleRecoveryDeadlineSeconds = 5.0f;
constexpr float kHurdleRecoveryBonus = 16.0f;

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

void AuditNoCadenceRewardState::reset(const CourseSpec& course) {
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
    earlyStartMilestoneSeen_ = false;
    smoothedTorsoY_ = 0.0f;
    torsoHeightAnchorReady_ = false;
    for (int foot = 0; foot < 2; ++foot) {
        footAirSeconds_[foot] = 0.0f;
        footGroundSeconds_[foot] = 0.0f;
        footLiftConfirmed_[foot] = false;
    }
    lastLandingFoot_ = -1;
    lastLandingTime_ = 0.0f;
    lastLandingProgress_ = 0.0f;
    classicHurdleArmed_ = false;
    classicHurdleCrossed_ = false;
    classicHurdleJumpSeen_ = false;
    classicHurdleJumpClosed_ = false;
    classicHurdleLandingConfirmed_ = false;
    classicHurdleRecoverySeen_ = false;
    classicHurdleCrossTime_ = 0.0f;
    classicHurdleJumpTime_ = 0.0f;
    classicHurdleAirTime_ = 0.0f;
    classicHurdleGroundTime_ = 0.0f;
    longJumpLandingSeen_ = false;
    longJumpPostLandingX_ = 0.0f;
}

float AuditNoCadenceRewardState::calculate(
    RewardProfile profile,
    const CourseSpec& course,
    const RgFrame& frame,
    bool truncated) {
    switch (profile) {
    case RewardProfile::Classic100mTask: {
        // Task-only ablation: leave angle, cadence and vertical excursion to
        // the policy. Count each new metre once; pure integrated speed would
        // pay the same total for every successful 100m trajectory.
        const float cappedProgress = std::clamp(frame.score, 0.0f, 100.0f);
        const float progress = std::max(0.0f, cappedProgress - progressScore_);
        progressScore_ = std::max(progressScore_, cappedProgress);
        const float elapsed = std::max(0.0f, frame.sim_time - previousRewardTime_);
        previousRewardTime_ = frame.sim_time;
        const float torsoHeight = kTrackY - frame.bodies[kTorsoBodySlot].y;
        const bool upright = torsoHeight >= kUprightHeightRatio * (kTrackY - kStandingBodyY);
        const float previousLowTime = lowPostureTime_;
        lowPostureTime_ = upright ? 0.0f : lowPostureTime_ + elapsed;
        const float lowSeconds = upright ? 0.0f : std::max(0.0f, lowPostureTime_ - kLowPostureGraceSeconds)
            - std::max(0.0f, previousLowTime - kLowPostureGraceSeconds);
        penalizedLowPostureSeconds_ += lowSeconds;
        float reward = (upright ? 2.0f * progress : 0.0f) - elapsed - 5.0f * lowSeconds;
        if (frame.failed || truncated) {
            reward -= 200.0f;
        } else if (frame.jump_landed) {
            if (penalizedLowPostureSeconds_ <= kMaximumLowPostureFinishFraction * frame.sim_time) {
                // Keep successful finishes positive throughout the course's
                // original 120s limit. Every second saved is worth 21 points
                // including the dense time cost, without a 43s reward plateau.
                reward += 2800.0f - 20.0f * frame.sim_time;
            } else {
                reward -= 200.0f;
            }
        }
        return reward;
    }
    case RewardProfile::Classic100m: {
        const float cappedProgress = std::clamp(frame.score, 0.0f, 100.0f);
        const float startProgress = std::max(0.0f,
            std::min(cappedProgress, kStartMilestoneMeters)
                - std::min(progressScore_, kStartMilestoneMeters));
        const float progressReward = std::max(0.0f, cappedProgress - progressScore_);
        const float earlyStartProgress = std::max(0.0f,
            std::min(cappedProgress, kEarlyStartMeters) - std::min(progressScore_, kEarlyStartMeters));
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
        bool hurdleJumpPhase = false;
        if (!course.hurdleLocations.empty()) {
            // Use the fixed course location; a knocked obstacle must not move
            // or reopen the reward window. No jump/high-body reward is added.
            const float hurdleX = course.hurdleLocations.front();
            const float hurdleDx = (torsoX - hurdleX) / kReferenceUnitsPerRaceMeter;
            const float maximumDx = (maximumTorsoX_ - hurdleX) / kReferenceUnitsPerRaceMeter;
            classicHurdleArmed_ |= maximumDx >= -kHurdleArmMeters;
            if (!classicHurdleCrossed_ && maximumDx >= 0.0f) {
                classicHurdleCrossed_ = true;
                classicHurdleCrossTime_ = frame.sim_time;
            }
            classicHurdleAirTime_ = !footSupported ? classicHurdleAirTime_ + elapsed : 0.0f;
            if (classicHurdleArmed_ && !classicHurdleJumpSeen_
                && hurdleDx >= -kHurdleJumpBeforeMeters && hurdleDx <= kHurdleJumpAfterMeters
                && !frame.failed && !truncated && upright
                && classicHurdleAirTime_ + 1e-5f >= kFootAirConfirmationSeconds
                && torsoHeight / standingHeight >= kHurdleJumpHeightRatio) {
                classicHurdleJumpSeen_ = true;
                classicHurdleJumpTime_ = frame.sim_time;
                classicHurdleLandingConfirmed_ = false;
                classicHurdleGroundTime_ = 0.0f;
            }
            classicHurdleGroundTime_ = footSupported && hurdleDx >= kHurdleLandingAfterMeters
                ? classicHurdleGroundTime_ + elapsed : 0.0f;
            if (classicHurdleGroundTime_ + 1e-5f >= kFootGroundConfirmationSeconds) {
                classicHurdleLandingConfirmed_ = true;
            }
            if (classicHurdleJumpSeen_ && !classicHurdleJumpClosed_) {
                const bool landed = classicHurdleGroundTime_ + 1e-5f >= kFootGroundConfirmationSeconds;
                if (landed || maximumDx >= kHurdlePhaseEndMeters
                    || frame.sim_time - classicHurdleJumpTime_ >= kHurdleMaximumJumpSeconds) {
                    classicHurdleJumpClosed_ = true;
                    if (landed) smoothedTorsoY_ = frame.bodies[kTorsoBodySlot].y;
                } else {
                    hurdleJumpPhase = true;
                }
            }
            // Consume the first recovery crossing even if it is unqualified;
            // going back and crossing again can never claim another bonus.
            if (!classicHurdleRecoverySeen_ && maximumDx >= kHurdleRecoveryMeters) {
                classicHurdleRecoverySeen_ = true;
                if (classicHurdleCrossed_ && classicHurdleLandingConfirmed_ && upright
                    && !frame.failed && !truncated && paceMeasurementReady_
                    && penalizedLowPostureSeconds_ <= kMaximumLowPostureFinishFraction * frame.sim_time) {
                    const float duration = frame.sim_time - classicHurdleCrossTime_;
                    const float timeQuality = std::clamp((kHurdleRecoveryDeadlineSeconds - duration)
                        / (kHurdleRecoveryDeadlineSeconds - kHurdleRecoveryTargetSeconds), 0.0f, 1.0f);
                    const float speedQuality = std::clamp((recentProgressSpeed_ - kSpeedRewardMinimum)
                        / (kTargetProgressSpeed - kSpeedRewardMinimum), 0.0f, 1.0f);
                    reward += kHurdleRecoveryBonus * timeQuality * speedQuality;
                }
            }
        }
        // Soft posture feedback also applies during flight, not only at the
        // instants of supported progress. Leave the first 2s for acceleration.
        if (upright && frame.sim_time >= 2.0f) {
            const float leanExcess = std::clamp(
                (std::abs(leanDegrees - kLeanTargetMinDegrees) - kLeanPostureGraceDegrees)
                    / kLeanPostureToleranceDegrees, 0.0f, 1.0f);
            reward -= kLeanPosturePenaltyPerSecond * (hurdleJumpPhase ? 0.5f : 1.0f)
                * elapsed * leanExcess * leanExcess;
        }
        const float torsoY = frame.bodies[kTorsoBodySlot].y;
        // Torso height is a proxy: the ABI does not expose body masses for COM.
        if (!torsoHeightAnchorReady_) {
            smoothedTorsoY_ = torsoY;
        } else if (elapsed > 0.0f) {
            smoothedTorsoY_ += (torsoY - smoothedTorsoY_) * elapsed / (kHeightSmoothingSeconds + elapsed);
            if (upright && frame.sim_time >= 2.0f && !hurdleJumpPhase) {
                const float oscillation = std::abs(torsoY - smoothedTorsoY_) / standingHeight;
                const float excess = std::clamp((oscillation - kHeightOscillationGrace) / kHeightOscillationTolerance, 0.0f, 1.0f);
                reward -= kVerticalMotionPenaltyPerSecond * elapsed * excess * excess;
            }
        }
        torsoHeightAnchorReady_ = true;
        bool landings[2] = {false, false};
        const bool contacts[2] = {frame.left_foot_grounded != 0, frame.right_foot_grounded != 0};
        for (int foot = 0; foot < 2; ++foot) {
            if (!contacts[foot]) {
                footAirSeconds_[foot] += elapsed;
                footGroundSeconds_[foot] = 0.0f;
                footLiftConfirmed_[foot] |= footAirSeconds_[foot] + 1e-5f >= kFootAirConfirmationSeconds;
            } else {
                footAirSeconds_[foot] = 0.0f;
                footGroundSeconds_[foot] += elapsed;
                if (footLiftConfirmed_[foot] && footGroundSeconds_[foot] + 1e-5f >= kFootGroundConfirmationSeconds) {
                    landings[foot] = true;
                    footLiftConfirmed_[foot] = false;
                }
            }
        }
        if (landings[0] != landings[1]) {
            const int foot = landings[0] ? 0 : 1;
            const float interval = frame.sim_time - lastLandingTime_;
            if (foot != lastLandingFoot_ && interval + 1e-6f >= kMinimumLandingIntervalSeconds) {
                const float stepProgress = std::clamp(progressScore_ - lastLandingProgress_, 0.0f, 5.0f);
                if (lastLandingFoot_ >= 0 && upright && !hurdleJumpPhase && !frame.failed && !truncated
                    && leanDegrees >= kBackwardLeanLimitDegrees && leanDegrees <= 30.0f
                    && paceMeasurementReady_ && recentProgressSpeed_ >= kSpeedRewardMinimum && stepProgress >= 0.2f) {
                    const float cadenceQuality = std::clamp((1.0f / interval - kCadenceBaseline) / (kCadenceTarget - kCadenceBaseline), 0.0f, 1.0f);
                    reward += kCadenceProgressBonus * stepProgress * cadenceQuality;
                }
                lastLandingFoot_ = foot;
                lastLandingTime_ = frame.sim_time;
                lastLandingProgress_ = progressScore_;
            }
        }
        if (upright) {
            reward += kProgressRewardScale * progressReward;
        }
        if (upright && footSupported) {
            reward += kProgressRewardScale * kSupportedProgressBonus * progressReward;
            const float targetError = std::max({kLeanTargetMinDegrees - leanDegrees,
                leanDegrees - kLeanTargetMaxDegrees, 0.0f});
            const float leanQuality = std::clamp(1.0f - targetError / kLeanToleranceDegrees, 0.0f, 1.0f);
            reward += kForwardLeanProgressBonus * progressReward * leanQuality;
        }
        if (upright && leanDegrees >= kBackwardLeanLimitDegrees && leanDegrees <= 30.0f) {
            // Extend speed feedback only where the optional flat actor learns;
            // acceleration, hurdle recovery and deficit penalties keep v31 goals.
            const bool flatProgress = (cappedProgress >= 10.0f && cappedProgress < 45.0f)
                || (cappedProgress >= 65.0f && cappedProgress <= 100.0f);
            const float speedTarget = flatProgress ? kFlatTargetProgressSpeed : kTargetProgressSpeed;
            const float speedBonus = flatProgress ? kFlatSpeedProgressBonus : kSpeedProgressBonus;
            const float speedQuality = std::clamp(
                (recentProgressSpeed_ - kSpeedRewardMinimum) / (speedTarget - kSpeedRewardMinimum), 0.0f, 1.0f);
            reward += speedBonus * progressReward * speedQuality;
            // Give acceleration feedback before crossing 10m. Reward only new
            // distance, taper with game time, and cap the total at 40 points.
            if (!frame.failed && !truncated) {
                const float startProgressQuality = std::clamp(
                    (kStartProgressDeadlineSeconds - frame.sim_time)
                        / (kStartProgressDeadlineSeconds - kStartTargetSeconds), 0.0f, 1.0f);
                reward += kStartProgressBonusPerMeter * startProgress * startProgressQuality;
                // Unlike the old 10m term, early progress has no flat time
                // plateau: earlier new distance always gets more feedback.
                const float earlyQuality = std::clamp(
                    (kEarlyStartDeadlineSeconds - frame.sim_time) / kEarlyStartDeadlineSeconds, 0.0f, 1.0f);
                reward += kEarlyStartProgressBonusPerMeter * earlyStartProgress * earlyQuality;
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
        if (!earlyStartMilestoneSeen_ && cappedProgress >= kEarlyStartMeters) {
            earlyStartMilestoneSeen_ = true;
            if (upright && !frame.failed && !truncated
                && leanDegrees >= kBackwardLeanLimitDegrees && leanDegrees <= 30.0f
                && penalizedLowPostureSeconds_ <= kMaximumLowPostureFinishFraction * frame.sim_time) {
                const float quality = std::clamp((kEarlyStartDeadlineSeconds - frame.sim_time)
                    / (kEarlyStartDeadlineSeconds - kEarlyStartTargetSeconds), 0.0f, 1.0f);
                reward += kEarlyStartBonus * quality;
            }
        }
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
