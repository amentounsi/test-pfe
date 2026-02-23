/**
 * CardDetector.cpp
 * Modular 7-stage Tunisian CIN detection
 *
 * Stage 1 – preprocessFrame     : gray → BilateralFilter → adaptive Canny → morphClose → dilate
 * Stage 2 – extractContours     : findContours, sort desc area, top-N
 * Stage 3 – rankContours        : approxPolyDP, 4-vertex, convex, area, ratio, edge density
 * Stage 4 – selectBestCandidate : weighted score
 * Stage 5 – validateFinal       : score > threshold
 * Stage 6 – validateRedCorners  : Cr channel check in all 4 corners (any orientation)
 * Stage 7 – temporalBuffer      : require 4/5 consecutive valid frames
 */

#include "CardDetector.h"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <android/log.h>

#define LOG_TAG "CardDetector"
#define LOGD(...) __android_log_print(ANDROID_LOG_INFO,  LOG_TAG, __VA_ARGS__)
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,  LOG_TAG, __VA_ARGS__)

namespace CardDetection {

// ──────────────────────────────────────────────
// Construction
// ──────────────────────────────────────────────

CardDetector::CardDetector()  : config_() {}
CardDetector::CardDetector(const DetectionConfig& cfg) : config_(cfg) {}
CardDetector::~CardDetector() = default;

void CardDetector::setConfig(const DetectionConfig& cfg) { config_ = cfg; }
DetectionConfig CardDetector::getConfig() const { return config_; }

void CardDetector::setCrMat(const cv::Mat& cr) {
    crMat_ = cr;  // shallow copy (or empty if not available)
}

// ──────────────────────────────────────────────
// Main pipeline
// ──────────────────────────────────────────────

CardDetectionResult CardDetector::detectCard(const cv::Mat& frame) {
    CardDetectionResult result;
    if (frame.empty()) { LOGD("detectCard: empty frame"); return result; }

    const int origW = frame.cols;
    const int origH = frame.rows;

    // ── Downscale for processing ──
    float scale = 1.f;
    cv::Mat procFrame;
    cv::Mat procCr;
    if (config_.processWidth > 0 && origW > config_.processWidth) {
        scale = static_cast<float>(config_.processWidth) / origW;
        int newH = static_cast<int>(origH * scale);
        cv::resize(frame, procFrame, cv::Size(config_.processWidth, newH), 0, 0, cv::INTER_AREA);
        // Resize Cr mat to the same processed dimensions if available
        if (!crMat_.empty()) {
            cv::resize(crMat_, procCr, cv::Size(config_.processWidth, newH), 0, 0, cv::INTER_AREA);
        }
    } else {
        procFrame = frame;
        procCr    = crMat_;
    }

    const int W = procFrame.cols;
    const int H = procFrame.rows;
    const double imageArea = static_cast<double>(W) * H;

    LOGD("detectCard: orig %dx%d → proc %dx%d  scale=%.3f  hasCr=%d",
         origW, origH, W, H, scale, procCr.empty() ? 0 : 1);

    // Stage 1
    cv::Mat edges = preprocessFrame(procFrame);

    if (config_.debugMode) {
        result.debug.edgeWhitePixels = cv::countNonZero(edges);
        LOGD("Stage1: edgeWhitePixels=%d", result.debug.edgeWhitePixels);
    }

    // Stage 2
    auto contours = extractContours(edges);
    result.debug.totalContours = static_cast<int>(contours.size());
    result.debug.topNContours  = static_cast<int>(contours.size());
    LOGD("Stage2: kept topN=%d", result.debug.topNContours);

    if (!contours.empty()) {
        double largestArea = cv::contourArea(contours[0]);
        result.debug.largestContourAreaRatio = static_cast<float>(largestArea / imageArea);
        LOGD("Stage2: largest ratio=%.4f", result.debug.largestContourAreaRatio);
    }

    // Stage 7 – temporal buffer: invalidate slot on early reject
    if (contours.empty()) {
        if (static_cast<int>(temporalBuf_.size()) < config_.temporalBufferSize)
            temporalBuf_.resize(config_.temporalBufferSize);
        temporalBuf_[temporalIdx_ % config_.temporalBufferSize].isValid = false;
        temporalIdx_++;
        return result;
    }

    // Stage 3
    auto candidates = rankContours(contours, imageArea, W, H, cannyEdges_, &result.debug);
    result.debug.candidateQuads = static_cast<int>(candidates.size());
    LOGD("Stage3: candidateQuads=%d", result.debug.candidateQuads);

    // Stage 7 – temporal buffer: invalidate slot on Stage 3 reject
    if (candidates.empty()) {
        if (static_cast<int>(temporalBuf_.size()) < config_.temporalBufferSize)
            temporalBuf_.resize(config_.temporalBufferSize);
        temporalBuf_[temporalIdx_ % config_.temporalBufferSize].isValid = false;
        temporalIdx_++;
        return result;
    }

    // Stage 4
    ContourCandidate best = selectBestCandidate(candidates);
    result.debug.bestScore  = best.score;
    result.debug.candidates = candidates;
    LOGD("Stage4: bestScore=%.3f  areaRatio=%.3f  aspect=%.3f  rect=%.3f",
         best.score, best.areaRatio, best.aspectRatio, best.rectangularity);

    // Stage 5 – geometric validation
    bool geomOk = (best.score   >= config_.minScore &&
                   best.areaRatio >= config_.minAreaRatio &&
                   best.areaRatio <= config_.maxAreaRatio);

    if (!geomOk) {
        LOGD("Stage5: REJECTED geometry  score=%.3f areaRatio=%.3f", best.score, best.areaRatio);
        if (static_cast<int>(temporalBuf_.size()) < config_.temporalBufferSize)
            temporalBuf_.resize(config_.temporalBufferSize);
        temporalBuf_[temporalIdx_ % config_.temporalBufferSize].isValid = false;
        temporalIdx_++;
        return result;
    }
    LOGD("Stage5: geometry OK");

    // Stage 6 – red corner validation
    float redScore = 0.f;
    if (config_.redValidationEnabled && !procCr.empty()) {
        redScore = validateRedCorners(best.quad, procCr, W, H);
        result.debug.redScore     = redScore;
        result.debug.redValidated = (redScore > 0.f);
        LOGD("Stage6: redScore=%.3f  validated=%d", redScore, result.debug.redValidated ? 1 : 0);
        if (redScore == 0.f) {
            LOGD("Stage6: REJECTED by red check");
            if (static_cast<int>(temporalBuf_.size()) < config_.temporalBufferSize)
                temporalBuf_.resize(config_.temporalBufferSize);
            temporalBuf_[temporalIdx_ % config_.temporalBufferSize].isValid = false;
            temporalIdx_++;
            return result;
        }
    } else if (config_.redValidationEnabled && procCr.empty()) {
        LOGD("Stage6: SKIPPED (no Cr mat)");
    }

    // Compute final confidence incorporating red score
    float confidence = best.score;
    if (redScore > 0.f) confidence = best.score * 0.85f + redScore * 0.15f;

    // Build validated result (corners in original frame coordinates)
    auto corners = sortCorners(best.quad);
    if (scale != 1.f) {
        float inv = 1.f / scale;
        for (auto& c : corners) { c.x *= inv; c.y *= inv; }
    }

    // Stage 7 – temporal buffer
    if (static_cast<int>(temporalBuf_.size()) < config_.temporalBufferSize)
        temporalBuf_.resize(config_.temporalBufferSize);

    auto& entry     = temporalBuf_[temporalIdx_ % config_.temporalBufferSize];
    entry.isValid   = true;
    entry.score     = confidence;
    entry.corners   = corners;
    temporalIdx_++;

    int validCount = 0;
    for (auto& e : temporalBuf_) if (e.isValid) validCount++;
    result.debug.temporalValidCount = validCount;
    LOGD("Stage7: temporalValid=%d/%d", validCount, config_.temporalBufferSize);

    if (validCount >= config_.temporalMinValid) {
        // Use latest valid result (more responsive than averaging)
        result.isValid    = true;
        result.confidence = confidence;
        result.corners    = corners;
        LOGI("DETECTED  score=%.2f  confidence=%.2f  redScore=%.2f",
             best.score, confidence, redScore);
    } else {
        LOGD("Stage7: waiting for temporal confirmation (%d/%d)",
             validCount, config_.temporalMinValid);
    }

    return result;
}

// ══════════════════════════════════════════════
// STAGE 1  –  Preprocessing
// ══════════════════════════════════════════════

cv::Mat CardDetector::preprocessFrame(const cv::Mat& input) {
    // 1-a  Ensure grayscale (input should already be gray from JNI)
    if (input.channels() == 4)
        cv::cvtColor(input, gray_, cv::COLOR_BGRA2GRAY);
    else if (input.channels() == 3)
        cv::cvtColor(input, gray_, cv::COLOR_BGR2GRAY);
    else
        gray_ = input;  // already gray — no copy needed

    // Diagnostic
    {
        cv::Scalar m, s;
        cv::meanStdDev(gray_, m, s);
        LOGD("Stage1-diag: gray %dx%d  mean=%.1f  stddev=%.1f",
             gray_.cols, gray_.rows, m[0], s[0]);
    }

    // 1-b  BilateralFilter – preserves edges, removes noise inside card
    //      Works on grayscale directly.
    cv::bilateralFilter(gray_, blurred_, config_.bilateralD,
                        config_.bilateralSigmaColor, config_.bilateralSigmaSpace);

    // 1-c  Adaptive Canny – thresholds derived from blurred image median
    //      Calculate median intensity
    cv::Mat flat;
    blurred_.reshape(1, 1).copyTo(flat);
    std::sort(flat.begin<uchar>(), flat.end<uchar>());
    double med = static_cast<double>(flat.at<uchar>(flat.total() / 2));
    int cannyLow  = std::max(10, static_cast<int>(med * config_.cannyMedianLow));
    int cannyHigh = std::min(250, static_cast<int>(med * config_.cannyMedianHigh));
    LOGD("Stage1: adaptiveCanny  median=%.0f  low=%d  high=%d", med, cannyLow, cannyHigh);

    cv::Canny(blurred_, edges_, cannyLow, cannyHigh);

    // Save raw Canny BEFORE morphological ops (for edge density check in Stage 3)
    cannyEdges_ = edges_.clone();

    // 1-d  Morphological Closing (bridge broken edges at card border)
    int m = config_.morphCloseSize | 1;
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(m, m));
    cv::morphologyEx(edges_, edges_, cv::MORPH_CLOSE, kernel);

    // 1-e  Dilate (connect last gaps so findContours gets a closed outline)
    if (config_.dilateSize > 0) {
        int dk = config_.dilateSize | 1;
        cv::Mat dk_kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(dk, dk));
        cv::dilate(edges_, edges_, dk_kernel);
    }

    return edges_.clone();
}

// ══════════════════════════════════════════════
// STAGE 2  –  Contour Extraction
// ══════════════════════════════════════════════

std::vector<std::vector<cv::Point>>
CardDetector::extractContours(const cv::Mat& edges) {
    std::vector<std::vector<cv::Point>> all;
    cv::findContours(edges.clone(), all, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    int totalFound = static_cast<int>(all.size());

    // Sort descending by area
    std::sort(all.begin(), all.end(), [](const auto& a, const auto& b) {
        return cv::contourArea(a) > cv::contourArea(b);
    });

    // Keep top-N only
    int keep = std::min(static_cast<int>(all.size()), config_.topN);
    all.resize(keep);

    if (config_.debugMode && !all.empty()) {
        for (int i = 0; i < keep; i++) {
            double a = cv::contourArea(all[i]);
            LOGD("Stage2: contour[%d] area=%.0f", i, a);
        }
    }
    LOGD("Stage2: found %d total, keeping top %d", totalFound, keep);

    return all;
}

// ══════════════════════════════════════════════
// STAGE 3  –  Geometric Ranking
// ══════════════════════════════════════════════

std::vector<ContourCandidate>
CardDetector::rankContours(
    const std::vector<std::vector<cv::Point>>& contours,
    double imageArea, int imgW, int imgH,
    const cv::Mat& cannyEdges,
    DebugInfo* debugOut)
{
    std::vector<ContourCandidate> out;
    int rejArea = 0, rejApprox = 0, rejAspect = 0, rejEdgeDensity = 0;

    for (size_t i = 0; i < contours.size(); i++) {
        const auto& c = contours[i];
        double area = cv::contourArea(c);
        double areaRatio = area / imageArea;

        // Area gate
        if (areaRatio < config_.minAreaRatio || areaRatio > config_.maxAreaRatio) {
            LOGD("Stage3: contour[%zu] areaRatio=%.3f SKIP (range %.3f-%.3f)",
                 i, areaRatio, config_.minAreaRatio, config_.maxAreaRatio);
            rejArea++;
            continue;
        }

        // Try approxPolyDP with several epsilon values
        std::vector<cv::Point> bestApprox;
        double bestApproxArea = -1;

        for (double eps = 0.01; eps <= 0.06; eps += 0.005) {
            std::vector<cv::Point> approx;
            double epsilon = eps * cv::arcLength(c, true);
            cv::approxPolyDP(c, approx, epsilon, true);

            if (approx.size() == 4 && cv::isContourConvex(approx)) {
                double a = cv::contourArea(approx);
                if (a > bestApproxArea) {
                    bestApproxArea = a;
                    bestApprox     = approx;
                }
            }
        }

        // Fallback: convex hull → approx
        if (bestApprox.empty()) {
            std::vector<cv::Point> hull;
            cv::convexHull(c, hull);
            for (double eps = 0.02; eps <= 0.06; eps += 0.005) {
                std::vector<cv::Point> approx;
                double epsilon = eps * cv::arcLength(hull, true);
                cv::approxPolyDP(hull, approx, epsilon, true);
                if (approx.size() == 4 && cv::isContourConvex(approx)) {
                    double a = cv::contourArea(approx);
                    if (a > bestApproxArea) {
                        bestApproxArea = a;
                        bestApprox     = approx;
                    }
                }
            }
        }

        if (bestApprox.empty()) {
            LOGD("Stage3: contour[%zu] no 4-vertex convex approx", i);
            rejApprox++;
            continue;
        }

        // Re-check area using the APPROXIMATED quad (can be larger than original contour)
        double approxAreaRatio = bestApproxArea / imageArea;
        if (approxAreaRatio < config_.minAreaRatio || approxAreaRatio > config_.maxAreaRatio) {
            LOGD("Stage3: contour[%zu] approxAreaRatio=%.3f SKIP (range %.3f-%.3f)",
                 i, approxAreaRatio, config_.minAreaRatio, config_.maxAreaRatio);
            rejArea++;
            continue;
        }

        // Check aspect ratio
        float aspectRatio = calcAspectRatio(bestApprox);
        float target      = config_.targetAspectRatio;
        float invTarget   = 1.f / target;
        float errL = std::abs(aspectRatio - target)    / target;
        float errP = std::abs(aspectRatio - invTarget)  / invTarget;
        float minErr = std::min(errL, errP);

        if (minErr > config_.aspectRatioTolerance) {
            LOGD("Stage3: contour[%zu] aspect=%.3f err=%.3f SKIP (tol=%.3f)",
                 i, aspectRatio, minErr, config_.aspectRatioTolerance);
            rejAspect++;
            continue;
        }

        // Edge density check: verify real Canny edges exist along quad borders
        float edgeDensity = calcEdgeDensity(bestApprox, cannyEdges);
        if (edgeDensity < config_.edgeDensityThreshold) {
            LOGD("Stage3: contour[%zu] edgeDensity=%.3f SKIP (min=%.3f)",
                 i, edgeDensity, config_.edgeDensityThreshold);
            rejEdgeDensity++;
            continue;
        }

        // Build candidate
        ContourCandidate cand;
        cand.quad           = bestApprox;
        cand.area           = bestApproxArea;
        cand.areaRatio      = approxAreaRatio;  // use approx area (consistent with Stage 5 check)
        cand.aspectRatio    = aspectRatio;
        cand.rectangularity = calcRectangularity(bestApprox);
        cand.centerDist     = calcCenterDistance(bestApprox, imgW, imgH);
        cand.edgeDensity    = edgeDensity;

        LOGD("Stage3: contour[%zu] CANDIDATE area=%.0f(%.1f%%)  aspect=%.3f  rect=%.3f  center=%.3f",
             i, cand.area, cand.areaRatio * 100, cand.aspectRatio,
             cand.rectangularity, cand.centerDist);

        out.push_back(cand);
    }

    LOGD("Stage3 summary: %zu candidates, rejArea=%d rejApprox=%d rejAspect=%d rejEdge=%d",
         out.size(), rejArea, rejApprox, rejAspect, rejEdgeDensity);

    if (debugOut) {
        debugOut->rejectedByArea   = rejArea;
        debugOut->rejectedByApprox = rejApprox;
        debugOut->rejectedByAspect = rejAspect;
        debugOut->rejectedByEdgeDensity = rejEdgeDensity;
    }

    return out;
}

// ══════════════════════════════════════════════
// STAGE 4  –  Multi-criteria Scoring
// ══════════════════════════════════════════════

ContourCandidate CardDetector::selectBestCandidate(
    std::vector<ContourCandidate>& candidates)
{
    // Score = w_area * AreaScore + w_ratio * RatioScore
    //       + w_rect * RectangularityScore + w_center * CenterScore

    for (auto& c : candidates) {
        // AreaScore: larger is better, normalised [0..1]
        float areaScore = static_cast<float>(
            std::min(c.areaRatio / 0.30, 1.0));  // 30 % of frame = perfect

        // RatioScore: closeness to target, 1.0 = perfect
        float target    = config_.targetAspectRatio;
        float invTarget = 1.f / target;
        float errL = std::abs(c.aspectRatio - target)    / target;
        float errP = std::abs(c.aspectRatio - invTarget)  / invTarget;
        float ratioScore = 1.f - std::min(errL, errP);
        ratioScore = std::max(0.f, ratioScore);

        // RectangularityScore: already in [0..1]
        float rectScore = c.rectangularity;

        // EdgeDensityScore: already in [0..1]
        float edgeDensityScore = c.edgeDensity;

        // CenterScore: closer to center is better
        float centerScore = 1.f - c.centerDist;
        centerScore = std::max(0.f, centerScore);

        c.score = config_.wArea   * areaScore
                + config_.wRatio  * ratioScore
                + config_.wRectangularity * rectScore
                + config_.wEdgeDensity * edgeDensityScore
                + config_.wCenter * centerScore;

        LOGD("Stage4: score=%.3f  [area=%.2f  ratio=%.2f  rect=%.2f  edge=%.2f  center=%.2f]",
             c.score, areaScore, ratioScore, rectScore, edgeDensityScore, centerScore);
    }

    // Sort descending by score
    std::sort(candidates.begin(), candidates.end(),
              [](const auto& a, const auto& b) { return a.score > b.score; });

    return candidates.front();
}

// ══════════════════════════════════════════════
// Helpers
// ══════════════════════════════════════════════

/**
 * Edge density: for each side of the quad, sample N points and check if
 * real Canny edge pixels exist nearby. Returns average density across all 4 sides.
 * A real card border → high density (edges on all 4 sides).
 * A desk/door blob  → low density (edges on 1–2 sides only).
 */
float CardDetector::calcEdgeDensity(
    const std::vector<cv::Point>& quad, const cv::Mat& cannyEdges)
{
    if (quad.size() != 4 || cannyEdges.empty()) return 0.f;

    const int samplesPerSide = 20;
    const int radius = 3;   // search ±3 pixels around each sample point
    float sideDensities[4];

    for (int s = 0; s < 4; s++) {
        cv::Point p1 = quad[s];
        cv::Point p2 = quad[(s + 1) % 4];
        int edgeCount = 0;

        for (int i = 0; i <= samplesPerSide; i++) {
            float t = static_cast<float>(i) / samplesPerSide;
            int x = static_cast<int>(p1.x + t * (p2.x - p1.x));
            int y = static_cast<int>(p1.y + t * (p2.y - p1.y));

            bool hasEdge = false;
            for (int dy = -radius; dy <= radius && !hasEdge; dy++) {
                for (int dx = -radius; dx <= radius && !hasEdge; dx++) {
                    int nx = x + dx, ny = y + dy;
                    if (nx >= 0 && nx < cannyEdges.cols &&
                        ny >= 0 && ny < cannyEdges.rows) {
                        if (cannyEdges.at<uchar>(ny, nx) > 0) hasEdge = true;
                    }
                }
            }
            if (hasEdge) edgeCount++;
        }
        sideDensities[s] = static_cast<float>(edgeCount) / (samplesPerSide + 1);
    }

    LOGD("  edgeDensity sides: [%.2f, %.2f, %.2f, %.2f]",
         sideDensities[0], sideDensities[1], sideDensities[2], sideDensities[3]);

    // Sort ascending: sideDensities[0]=worst, [3]=best
    std::sort(sideDensities, sideDensities + 4);

    // Return second-lowest: at least 3 of 4 sides must have good edges.
    // This tolerates one weak side (partial occlusion / shadow) while
    // rejecting desks/doors that only have 1-2 edges.
    return sideDensities[1];
}

// ══════════════════════════════════════════════
// STAGE 6  –  Red Corner Validation
// ══════════════════════════════════════════════

/**
 * Check all 4 corners of the quad for the red drapeau tunisien.
 * Uses the Cr (V) channel from YUV: Cr > threshold → red-ish.
 * Works in all card orientations – checks every corner, passes if ANY one has red.
 * Returns 1.0 (strict match), 0.6 (loose match), or 0.0 (no red found).
 */
float CardDetector::validateRedCorners(
    const std::vector<cv::Point>& quad,
    const cv::Mat& crMat,
    int procW, int procH)
{
    if (quad.size() != 4 || crMat.empty()) return 0.f;

    // Sort corners: TL, TR, BR, BL
    auto sortedPts = sortCorners(quad);

    // crMat is at procW x procH (same scale as processed gray)
    float qW = 0.f, qH = 0.f;
    // Estimate quad width/height from sorted corners
    qW = std::max({std::abs(sortedPts[1].x - sortedPts[0].x),
                   std::abs(sortedPts[2].x - sortedPts[3].x), 1.f});
    qH = std::max({std::abs(sortedPts[3].y - sortedPts[0].y),
                   std::abs(sortedPts[2].y - sortedPts[1].y), 1.f});

    int zoneW = std::max(4, static_cast<int>(qW * config_.redCornerZoneW));
    int zoneH = std::max(4, static_cast<int>(qH * config_.redCornerZoneH));

    float bestStrict = 0.f, bestLoose = 0.f;

    for (int ci = 0; ci < 4; ci++) {
        int cx = static_cast<int>(sortedPts[ci].x);
        int cy = static_cast<int>(sortedPts[ci].y);

        // Each corner extends INWARD:
        // ci=0 TL → extend right + down
        // ci=1 TR → extend left  + down
        // ci=2 BR → extend left  + up
        // ci=3 BL → extend right + up
        int x0, y0;
        switch (ci) {
            case 0: x0 = cx;         y0 = cy;         break; // TL
            case 1: x0 = cx - zoneW; y0 = cy;         break; // TR
            case 2: x0 = cx - zoneW; y0 = cy - zoneH; break; // BR
            default: x0 = cx;        y0 = cy - zoneH; break; // BL
        }
        // Clamp to crMat bounds
        x0 = std::max(0, std::min(x0, crMat.cols - 1));
        y0 = std::max(0, std::min(y0, crMat.rows - 1));
        int x1 = std::min(x0 + zoneW, crMat.cols);
        int y1 = std::min(y0 + zoneH, crMat.rows);
        if (x1 <= x0 || y1 <= y0) continue;

        cv::Rect roi(x0, y0, x1 - x0, y1 - y0);
        cv::Mat zone = crMat(roi);
        int total = zone.rows * zone.cols;
        if (total == 0) continue;

        // Count pixels above thresholds
        int cntStrict = cv::countNonZero(zone > config_.redCrThresholdStrict);
        int cntLoose  = cv::countNonZero(zone > config_.redCrThresholdLoose);
        float rS = static_cast<float>(cntStrict) / total;
        float rL = static_cast<float>(cntLoose)  / total;

        LOGD("  RedCorner[%d] zone(%d,%d %dx%d) strict=%.3f loose=%.3f",
             ci, x0, y0, x1-x0, y1-y0, rS, rL);

        bestStrict = std::max(bestStrict, rS);
        bestLoose  = std::max(bestLoose,  rL);
    }

    if (bestStrict >= config_.redMinRatioStrict) {
        LOGD("  RedCorner: CONFIRMED STRICT (%.3f)", bestStrict);
        return 1.0f;
    }
    if (bestLoose >= config_.redMinRatioLoose) {
        LOGD("  RedCorner: CONFIRMED LOOSE (%.3f)", bestLoose);
        return 0.6f;
    }
    LOGD("  RedCorner: REJECTED best strict=%.3f loose=%.3f (min=%.3f/%.3f)",
         bestStrict, bestLoose,
         config_.redMinRatioStrict, config_.redMinRatioLoose);
    return 0.f;
}

// ══════════════════════════════════════════════
// Helpers
// ══════════════════════════════════════════════

float CardDetector::ptDist(const cv::Point& a, const cv::Point& b) {
    float dx = static_cast<float>(b.x - a.x);
    float dy = static_cast<float>(b.y - a.y);
    return std::sqrt(dx * dx + dy * dy);
}

float CardDetector::calcAspectRatio(const std::vector<cv::Point>& quad) {
    if (quad.size() != 4) return 0.f;

    // Find centroid
    cv::Point cen(0, 0);
    for (auto& p : quad) { cen.x += p.x; cen.y += p.y; }
    cen.x /= 4; cen.y /= 4;

    std::vector<cv::Point> top, bot;
    for (auto& p : quad) {
        if (p.y < cen.y) top.push_back(p); else bot.push_back(p);
    }

    if (top.size() != 2 || bot.size() != 2) {
        cv::Rect r = cv::boundingRect(quad);
        return r.height == 0 ? 0.f : static_cast<float>(r.width) / r.height;
    }

    if (top[0].x > top[1].x) std::swap(top[0], top[1]);
    if (bot[0].x > bot[1].x) std::swap(bot[0], bot[1]);

    float w = (ptDist(top[0], top[1]) + ptDist(bot[0], bot[1])) / 2.f;
    float h = (ptDist(top[0], bot[0]) + ptDist(top[1], bot[1])) / 2.f;
    return h == 0 ? 0.f : w / h;
}

float CardDetector::calcRectangularity(const std::vector<cv::Point>& quad) {
    if (quad.size() != 4) return 0.f;

    // Compute cos of each interior angle; perfect rectangle → cos ≈ 0
    float totalDev = 0.f;
    for (int i = 0; i < 4; i++) {
        cv::Point A = quad[i];
        cv::Point B = quad[(i + 1) % 4];
        cv::Point C = quad[(i + 2) % 4];

        cv::Point v1 = A - B;
        cv::Point v2 = C - B;

        float dot   = static_cast<float>(v1.x * v2.x + v1.y * v2.y);
        float mag1  = std::sqrt(static_cast<float>(v1.x * v1.x + v1.y * v1.y));
        float mag2  = std::sqrt(static_cast<float>(v2.x * v2.x + v2.y * v2.y));

        if (mag1 < 1.f || mag2 < 1.f) return 0.f;

        float cosA  = dot / (mag1 * mag2);
        totalDev   += std::abs(cosA);   // ideal = 0
    }
    // Average |cos| across 4 angles.  0 → perfect rectangle, 1 → degenerate
    float avgDev = totalDev / 4.f;
    return 1.f - std::min(avgDev * 2.f, 1.f);   // scale & clamp to [0..1]
}

float CardDetector::calcCenterDistance(
    const std::vector<cv::Point>& quad, int imgW, int imgH)
{
    if (quad.size() != 4) return 1.f;

    float cx = 0, cy = 0;
    for (auto& p : quad) { cx += p.x; cy += p.y; }
    cx /= 4.f; cy /= 4.f;

    float dx = cx - imgW / 2.f;
    float dy = cy - imgH / 2.f;
    float dist = std::sqrt(dx * dx + dy * dy);
    float maxDist = std::sqrt(static_cast<float>(imgW * imgW + imgH * imgH)) / 2.f;
    return maxDist > 0 ? dist / maxDist : 1.f;
}

std::array<Point2D, 4> CardDetector::sortCorners(const std::vector<cv::Point>& quad) {
    std::array<Point2D, 4> corners;
    if (quad.size() != 4) return corners;

    // sum = x+y  →  TL has min sum,  BR has max sum
    // diff = x-y →  TR has max diff, BL has min diff
    int tl = 0, tr = 0, br = 0, bl = 0;
    int minS = INT_MAX, maxS = INT_MIN, minD = INT_MAX, maxD = INT_MIN;
    for (int i = 0; i < 4; i++) {
        int s = quad[i].x + quad[i].y;
        int d = quad[i].x - quad[i].y;
        if (s < minS) { minS = s; tl = i; }
        if (s > maxS) { maxS = s; br = i; }
        if (d < minD) { minD = d; bl = i; }
        if (d > maxD) { maxD = d; tr = i; }
    }

    corners[0] = Point2D(static_cast<float>(quad[tl].x), static_cast<float>(quad[tl].y));
    corners[1] = Point2D(static_cast<float>(quad[tr].x), static_cast<float>(quad[tr].y));
    corners[2] = Point2D(static_cast<float>(quad[br].x), static_cast<float>(quad[br].y));
    corners[3] = Point2D(static_cast<float>(quad[bl].x), static_cast<float>(quad[bl].y));
    return corners;
}

} // namespace CardDetection
