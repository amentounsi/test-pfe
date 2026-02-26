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
#include <chrono>

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
 * Detection state for hysteresis state machine
 * SEARCHING: No valid candidate, using strict entry thresholds
 * ALIGNING:  Candidate exists but not confirmed, using strict thresholds
 * LOCKED:    Confirmed detection, using relaxed keep thresholds
 */
enum class DetectionState {
    SEARCHING,
    ALIGNING,
    LOCKED
};

/**
 * Overlay bounds for guided detection (normalized 0-1)
 * When enabled, detection validates quads against this fixed frame
 * Includes hysteresis thresholds to prevent oscillation
 */
struct OverlayBounds {
    bool  enabled = false;   // Enable overlay-guided detection
    float x       = 0.f;     // Normalized 0-1 (left edge)
    float y       = 0.f;     // Normalized 0-1 (top edge)
    float width   = 0.f;     // Normalized 0-1
    float height  = 0.f;     // Normalized 0-1
    
    // Entry thresholds (strict - used in SEARCHING/ALIGNING states)
    float areaToleranceLow  = 0.25f;  // Quad must be >= 25% of overlay area
    float areaToleranceHigh = 2.20f;  // Quad must be <= 220% of overlay area
    float centerToleranceRatio = 0.35f; // Max distance from overlay center (35% of diagonal)
    float overlapMinRatio = 0.40f;    // Quad must overlap >= 40% with overlay
    
    // Keep thresholds (relaxed - used in LOCKED state for stability)
    float areaToleranceLowKeep  = 0.22f;  // Allow down to 22% when locked
    float areaToleranceHighKeep = 2.50f;  // Allow up to 250% when locked
    float centerToleranceRatioKeep = 0.40f; // Allow 40% center drift when locked
    float overlapMinRatioKeep = 0.35f;    // Allow 35% overlap when locked
};

/**
 * Intermediate candidate produced by Stage 3
 */
struct ContourCandidate {
    std::vector<cv::Point> quad;           // 4 corners
    double area                  = 0.0;
    double areaRatio             = 0.0;    // area / imageArea
    float  aspectRatio           = 0.f;
    float  rectangularity        = 0.f;    // how close angles are to 90 deg
    float  centerDist            = 0.f;    // normalised dist quad-centre to img-centre
    float  edgeDensity           = 0.f;    // fraction of border with real Canny edges
    float  borderContrastScore   = 0.f;    // inner/outer gradient across quad edges
    float  redScore              = 0.f;    // Stage 6: red corner validation score
    float  score                 = 0.f;    // composite geometry score (Stage 4)
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
    // --- Stage 0: Overlay-Guided Detection (NEW) ---
    OverlayBounds overlay;               // Fixed guide frame bounds (optional)
    bool useROICropping = true;          // Crop frame to overlay ROI before detection
    
    // --- Stage 1: preprocessing ---
    // CLAHE: local contrast enhancement (critical for card on light desk)
    double claheClipLimit    = 2.0;   // higher = more contrast boost
    int    claheTileSize     = 8;     // grid size (8x8 tiles at 480px)
    int   gaussianBlurSize   = 5;     // applied AFTER CLAHE
    // Adaptive Canny (calculated AFTER blur on blurred image)
    float cannyMedianLow       = 0.33f;  // Canny low  = median * 0.33
    float cannyMedianHigh      = 1.10f;  // Canny high = median * 1.10
    int   morphCloseSize       = 5;      // bridge broken edges at card border
    int   dilateSize           = 3;
    int   processWidth         = 480;    // 44% fewer pixels vs 640

    // --- Stage 2: contour extraction ---
    int topN                   = 8;

    // --- Stage 3: geometric filter ---
    float minAreaRatio         = 0.010f; // 1.0% min (card far from camera or on cluttered bg)
    float maxAreaRatio         = 0.20f;  // PHYSICAL CONSTRAINT: handheld CIN never exceeds 20% of frame
                                         // (screens/desks covering 30-40% → rejected)
    float targetAspectRatio    = 1.586f; // ID-1 standard (85.6 x 54 mm)
    float aspectRatioTolerance = 0.35f;  // ±35% → range 1.03-2.14 (handles tilted card + keyboard perspective)
    float edgeDensityThreshold = 0.20f;  // 2nd-worst side must have ≥20% real edges

    // --- Stage 4: scoring weights ---
    float wArea             = 0.20f;
    float wRatio            = 0.25f;
    float wRectangularity   = 0.25f;
    float wEdgeDensity      = 0.20f;
    float wCenter           = 0.10f;

    // --- Stage 5: final validation ---
    // minScore applies to FINAL confidence = geometry*0.5 + border*0.3 + red*0.2
    float minScore          = 0.65f;
    // Minimum geometry score floor (before combining with border/red)
    float minGeometryScore  = 0.50f;
    // BorderContrast normalization: contrast of 50 gray levels → score=1.0
    float borderContrastNorm = 50.f;

    // --- Stage 5e: appearance validation (semantic filter) ---
    // Rejects objects that pass geometry but are visually not CIN cards
    bool  appearanceValidationEnabled = true;
    int   appearanceWarpWidth  = 640;   // Perspective-correct ROI width
    int   appearanceWarpHeight = 400;   // Perspective-correct ROI height (ratio ≈ 1.6)
    float appearanceMeanMin    = 55.f;  // Reject if mean < 55 (very dark - no light at all)
    float appearanceStddevMax  = 55.f;  // Reject if stddev > 55 (too textured)
    float appearanceMeanLowLight = 85.f;  // Combined check threshold (lowered for low light)
    float appearanceStddevMedium = 50.f;  // Reject if mean < 85 AND stddev > 50

    // --- Stage 6: red corner validation ---
    // Checks all 4 corners; at least 1 must pass ALL 3 conditions.
    bool  redValidationEnabled   = true;
    int   redCrThreshold         = 145;   // Cr > 145 → red (Tunisian flag red is vivid)
    float redMinRatio            = 0.015f;// ≥1.5% of corner zone must be red (lowered for dim light)
    int   redWhiteYThreshold     = 180;   // Y > 180 → white pixel
    float redWhiteMinRatio       = 0.25f; // ≥25% of zone must be white (card background)
    float redClusterMinRatio     = 0.06f; // largest red contour ≥6% of zone (compact flag — blocks scattered PC icons)
    float redCornerZoneW         = 0.18f; // 18% of quad width per corner zone
    float redCornerZoneH         = 0.25f; // 25% of quad height per corner zone

    // --- Stage 7: temporal buffer ---
    int temporalBufferSize     = 6;     // keep last N frames
    int temporalMinValid       = 3;     // need M/N valid to confirm
    int lockedFailFramesToReset = 4;    // LOCKED → SEARCHING after N consecutive fails

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

    // Throttle: skip processing if called too soon (target 15 FPS max)
    std::chrono::steady_clock::time_point lastDetectionTime_;
    CardDetectionResult lastResult_;    // cached result for throttled frames
    static constexpr int kMinProcessIntervalMs = 66;  // 66ms = ~15 FPS

    // Temporal buffer (Stage 7)
    struct TemporalEntry {
        bool  isValid = false;
        float score   = 0.f;
        std::array<Point2D, 4> corners;
    };
    std::vector<TemporalEntry> temporalBuf_;
    int                        temporalIdx_ = 0;

    // Hysteresis state machine (Stage 7b)
    DetectionState detectionState_ = DetectionState::SEARCHING;
    int consecutiveFailFrames_ = 0;  // Counter for LOCKED→SEARCHING transition

    // Helpers
    float  calcEdgeDensity(const std::vector<cv::Point>& quad,
                           const cv::Mat& cannyEdges);
    /** Stage 6 – check Cr channel in all 4 corners for red flag */
    /** Stage 6 – check all 4 corners for red+white cluster (Tunisian flag pattern) */
    float  validateRedCorners(const std::vector<cv::Point>& quad,
                              const cv::Mat& crMat,
                              const cv::Mat& grayMat,
                              int procW, int procH);
    /** Measure inner/outer gray gradient across quad border sides → [0,1] */
    float  calcBorderContrast(const std::vector<cv::Point>& quad,
                              const cv::Mat& grayMat);
    float  calcAspectRatio(const std::vector<cv::Point>& quad);
    float  calcRectangularity(const std::vector<cv::Point>& quad);
    float  calcCenterDistance(const std::vector<cv::Point>& quad,
                              int imgW, int imgH);
    std::array<Point2D, 4> sortCorners(const std::vector<cv::Point>& quad);
    float  ptDist(const cv::Point& a, const cv::Point& b);
    
    // --- Stage 5e: Appearance Validation ---
    /** Validate quad appearance (mean luminance, texture variance)
     *  Returns true if quad passes appearance checks */
    bool   validateAppearance(const std::vector<cv::Point>& quad,
                               const cv::Mat& grayMat,
                               float& outMean, float& outStddev);
    
    // --- Overlay-Guided Detection (NEW) ---
    /** Check if quad satisfies overlay constraints (area, center, overlap)
     *  Uses hysteresis: LOCKED state uses relaxed keep-thresholds */
    bool   validateOverlayConstraints(const std::vector<cv::Point>& quad,
                                       int imgW, int imgH,
                                       const OverlayBounds& overlay,
                                       DetectionState state);
    /** Compute IoU (Intersection over Union) between quad and overlay rect */
    float  computeQuadOverlayOverlap(const std::vector<cv::Point>& quad,
                                      int imgW, int imgH,
                                      const OverlayBounds& overlay);
    /** Extract ROI from frame based on overlay bounds */
    cv::Mat extractOverlayROI(const cv::Mat& frame, 
                              const OverlayBounds& overlay,
                              cv::Rect& roiRect);
};

} // namespace CardDetection

#endif // CARD_DETECTOR_H
