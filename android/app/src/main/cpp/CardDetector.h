/**
 * CardDetector.h
 * Modular Tunisian ID Card (CIN) detection using OpenCV
 *
 * Architecture: 7 stages
 *   1. preprocessFrame     – gray → BilateralFilter → adaptive Canny → morphClose → dilate
 *   2. extractContours     – findContours, sort by area, keep top-N
 *   3. rankContours        – approxPolyDP, geometric filter + edge density
 *   4. selectBestCandidate – multi-criteria weighted scoring
 *   5. validateFinal       – score threshold
 *   6. validateRedCorners  – HSV/Cr red flag confirmation (all 4 orientations)
 *   7. temporalBuffer      – require N/M valid frames before reporting
 *
 * All tunables live in DetectionConfig below.
 */

#ifndef CARD_DETECTOR_H
#define CARD_DETECTOR_H

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>
#include <array>
#include <string>

namespace CardDetection {

// ──────────────────────────────────────────────
// Data structures
// ──────────────────────────────────────────────

struct Point2D {
    float x = 0.f;
    float y = 0.f;
    Point2D() = default;
    Point2D(float x_, float y_) : x(x_), y(y_) {}
};

/**
 * Intermediate candidate produced by Stage 3
 */
struct ContourCandidate {
    std::vector<cv::Point> quad;       // 4 corners
    double area            = 0.0;
    double areaRatio       = 0.0;      // area / imageArea
    float  aspectRatio     = 0.f;
    float  rectangularity  = 0.f;      // how close angles are to 90 deg
    float  centerDist      = 0.f;      // normalised dist quad-centre to img-centre
    float  edgeDensity     = 0.f;      // fraction of border with real Canny edges
    float  redScore        = 0.f;      // Stage 6: red corner validation score
    float  score           = 0.f;      // composite score (Stage 4)
};

/**
 * Debug information returned alongside detection result
 */
struct DebugInfo {
    int   totalContours    = 0;
    int   topNContours     = 0;
    int   candidateQuads   = 0;
    int   edgeWhitePixels  = 0;
    float bestScore        = 0.f;
    // Rejection counters (Stage 3)
    int   rejectedByArea        = 0;
    int   rejectedByApprox      = 0;
    int   rejectedByAspect      = 0;
    int   rejectedByEdgeDensity = 0;
    // Stage 6
    bool  redValidated          = false;
    float redScore              = 0.f;
    // Stage 7
    int   temporalValidCount    = 0;
    float largestContourAreaRatio = 0.f;
    std::vector<ContourCandidate> candidates;
};

/**
 * Final detection result
 */
struct CardDetectionResult {
    bool  isValid    = false;
    float confidence = 0.f;
    std::array<Point2D, 4> corners;   // TL, TR, BR, BL
    DebugInfo debug;

    CardDetectionResult() { corners.fill(Point2D()); }
};

// ──────────────────────────────────────────────
// Centralised configuration  (tune HERE only)
// ──────────────────────────────────────────────

struct DetectionConfig {
    // --- Stage 1: preprocessing ---
    // BilateralFilter (preserves edges better than GaussianBlur)
    int   bilateralD           = 5;      // small d=5 for performance; 7 on high-end
    float bilateralSigmaColor  = 20.f;   // keep LOW to preserve sharp edges
    float bilateralSigmaSpace  = 20.f;   // keep LOW to preserve sharp edges
    // Adaptive Canny (calculated AFTER blur on blurred image)
    float cannyMedianLow       = 0.33f;  // Canny low  = median * 0.33
    float cannyMedianHigh      = 1.10f;  // Canny high = median * 1.10
    int   morphCloseSize       = 7;      // lighter than before
    int   dilateSize           = 3;
    int   processWidth         = 640;

    // --- Stage 2: contour extraction ---
    int topN                   = 5;

    // --- Stage 3: geometric filter ---
    float minAreaRatio         = 0.02f;  // card far from camera
    float maxAreaRatio         = 0.85f;  // card very close to camera
    float targetAspectRatio    = 1.586f; // ID-1 standard (85.6 x 54 mm)
    float aspectRatioTolerance = 0.35f;  // wider for perspective distortion
    float edgeDensityThreshold = 0.20f;  // 2nd-worst side must have ≥20% real edges

    // --- Stage 4: scoring weights ---
    float wArea             = 0.20f;
    float wRatio            = 0.25f;
    float wRectangularity   = 0.25f;
    float wEdgeDensity      = 0.20f;
    float wCenter           = 0.10f;

    // --- Stage 5: final validation ---
    float minScore          = 0.45f;

    // --- Stage 6: red corner validation (Cr channel, works in all 4 orientations) ---
    bool  redValidationEnabled = true;
    int   redCrThresholdStrict = 145;   // Cr > 145 → definitely red
    int   redCrThresholdLoose  = 120;   // Cr > 120 → probably red
    float redMinRatioStrict    = 0.04f; // ≥4% of corner zone = confirmed
    float redMinRatioLoose     = 0.07f; // ≥7% of corner zone (loose pass)
    float redCornerZoneW       = 0.18f; // 18% of quad width per corner zone
    float redCornerZoneH       = 0.25f; // 25% of quad height per corner zone

    // --- Stage 7: temporal buffer ---
    int temporalBufferSize     = 5;     // keep last N frames
    int temporalMinValid       = 4;     // need M/N valid to confirm

    // --- Debug ---
    bool debugMode             = true;
};

// ──────────────────────────────────────────────
// Detector class
// ──────────────────────────────────────────────

class CardDetector {
public:
    CardDetector();
    explicit CardDetector(const DetectionConfig& cfg);
    ~CardDetector();

    /** Full pipeline – calls stages 1-7 */
    CardDetectionResult detectCard(const cv::Mat& grayFrame);

    /** Set the Cr (V-channel) plane BEFORE calling detectCard.
     *  Used for Stage 6 red validation. May be empty if not available. */
    void setCrMat(const cv::Mat& cr);  // must be called before detectCard

    void            setConfig(const DetectionConfig& cfg);
    DetectionConfig getConfig() const;

    // ── Public modular stages ──

    /** Stage 1 – gray → BilateralFilter → adaptive Canny → morphClose → dilate */
    cv::Mat preprocessFrame(const cv::Mat& grayInput);

    /** Stage 2 – findContours, sort desc by area, return top-N */
    std::vector<std::vector<cv::Point>> extractContours(const cv::Mat& edges);

    /** Stage 3 – approxPolyDP + geometric checks + edge density */
    std::vector<ContourCandidate> rankContours(
        const std::vector<std::vector<cv::Point>>& contours,
        double imageArea, int imageWidth, int imageHeight,
        const cv::Mat& cannyEdges,
        DebugInfo* debugOut = nullptr
    );

    /** Stage 4 – multi-criteria weighted scoring */
    ContourCandidate selectBestCandidate(
        std::vector<ContourCandidate>& candidates
    );

private:
    DetectionConfig config_;

    // Reusable mats
    cv::Mat gray_;
    cv::Mat blurred_;
    cv::Mat edges_;
    cv::Mat cannyEdges_;   // raw Canny before morphological ops
    cv::Mat crMat_;        // Cr (V) channel for red validation

    // Temporal buffer (Stage 7)
    struct TemporalEntry {
        bool  isValid = false;
        float score   = 0.f;
        std::array<Point2D, 4> corners;
    };
    std::vector<TemporalEntry> temporalBuf_;
    int                        temporalIdx_ = 0;

    // Helpers
    float  calcEdgeDensity(const std::vector<cv::Point>& quad,
                           const cv::Mat& cannyEdges);
    /** Stage 6 – check Cr channel in all 4 corners for red flag */
    float  validateRedCorners(const std::vector<cv::Point>& quad,
                              const cv::Mat& crMat,
                              int procW, int procH);
    float  calcAspectRatio(const std::vector<cv::Point>& quad);
    float  calcRectangularity(const std::vector<cv::Point>& quad);
    float  calcCenterDistance(const std::vector<cv::Point>& quad,
                              int imgW, int imgH);
    std::array<Point2D, 4> sortCorners(const std::vector<cv::Point>& quad);
    float  ptDist(const cv::Point& a, const cv::Point& b);
};

} // namespace CardDetection

#endif // CARD_DETECTOR_H
