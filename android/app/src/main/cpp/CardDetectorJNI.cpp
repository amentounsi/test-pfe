
/**
 * CardDetectorJNI.cpp
 * JNI bridge – exposes C++ CardDetector to Java/Kotlin
 *
 * Return format (float[21]):
 *   [0]  isValid          (1.0 / 0.0)
 *   [1]  confidence       (0..1)
 *   [2..9]  corners x0,y0 … x3,y3
 *   [10] edgeWhitePixels
 *   [11] totalContours
 *   [12] candidateQuads
 *   [13] bestScore
 *   [14] topNContours
 *   [15] rejectedByArea
 *   [16] rejectedByApprox
 *   [17] rejectedByAspect
 *   [18] largestContourAreaRatio
 *   [19] rejectedByEdgeDensity
 *   [20] temporalValidCount
 */

#include <jni.h>
#include <android/log.h>
#include <android/bitmap.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include "CardDetector.h"
#include <memory>

#define LOG_TAG "CardDetectorJNI"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,  LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

static constexpr int RESULT_LEN = 21;
static std::unique_ptr<CardDetection::CardDetector> g_detector = nullptr;

// Fill the 20-float result array from a CardDetectionResult
static void fillResult(float* out, const CardDetection::CardDetectionResult& r) {
    out[0] = r.isValid ? 1.f : 0.f;
    out[1] = r.confidence;
    if (r.isValid) {
        for (int i = 0; i < 4; i++) {
            out[2 + i * 2]     = r.corners[i].x;
            out[2 + i * 2 + 1] = r.corners[i].y;
        }
    }
    out[10] = static_cast<float>(r.debug.edgeWhitePixels);
    out[11] = static_cast<float>(r.debug.totalContours);
    out[12] = static_cast<float>(r.debug.candidateQuads);
    out[13] = r.debug.bestScore;
    out[14] = static_cast<float>(r.debug.topNContours);
    out[15] = static_cast<float>(r.debug.rejectedByArea);
    out[16] = static_cast<float>(r.debug.rejectedByApprox);
    out[17] = static_cast<float>(r.debug.rejectedByAspect);
    out[18] = r.debug.largestContourAreaRatio;
    out[19] = static_cast<float>(r.debug.rejectedByEdgeDensity);
    out[20] = static_cast<float>(r.debug.temporalValidCount);
}

extern "C" {

JNIEXPORT void JNICALL
Java_com_pfeprojet_carddetector_CardDetectorJNI_nativeInit(JNIEnv*, jclass) {
    if (!g_detector) {
        g_detector = std::make_unique<CardDetection::CardDetector>();
        LOGI("CardDetector initialised");
    }
}

JNIEXPORT void JNICALL
Java_com_pfeprojet_carddetector_CardDetectorJNI_nativeRelease(JNIEnv*, jclass) {
    g_detector.reset();
    LOGI("CardDetector released");
}

JNIEXPORT void JNICALL
Java_com_pfeprojet_carddetector_CardDetectorJNI_nativeSetConfig(
    JNIEnv*, jclass,
    jint cannyLow, jint cannyHigh, jint blurSize,
    jfloat minArea, jfloat maxArea,
    jfloat targetRatio, jfloat ratioTolerance)
{
    if (!g_detector) { LOGE("not initialised"); return; }

    CardDetection::DetectionConfig cfg;
    // cannyLow/High are now adaptive (ignored); blurSize maps to gaussianBlurSize
    (void)cannyLow; (void)cannyHigh;
    cfg.gaussianBlurSize   = (blurSize > 0 && blurSize <= 15) ? blurSize : 5;
    cfg.minAreaRatio       = minArea;
    cfg.maxAreaRatio       = maxArea;
    cfg.targetAspectRatio  = targetRatio;
    cfg.aspectRatioTolerance = ratioTolerance;
    g_detector->setConfig(cfg);
    LOGI("config updated");
}

JNIEXPORT void JNICALL
Java_com_pfeprojet_carddetector_CardDetectorJNI_nativeSetOverlay(
    JNIEnv*, jclass,
    jboolean enabled,
    jfloat x, jfloat y, jfloat width, jfloat height,
    jboolean useROICropping)
{
    if (!g_detector) { LOGE("not initialised"); return; }

    CardDetection::DetectionConfig cfg = g_detector->getConfig();
    cfg.overlay.enabled = enabled;
    cfg.overlay.x = x;
    cfg.overlay.y = y;
    cfg.overlay.width = width;
    cfg.overlay.height = height;
    cfg.useROICropping = useROICropping;
    
    g_detector->setConfig(cfg);
    LOGI("overlay config: enabled=%d [%.3f,%.3f %.3fx%.3f] useROI=%d",
         enabled, x, y, width, height, useROICropping);
}

// ── Detect from YUV (VisionCamera path) ──

JNIEXPORT jfloatArray JNICALL
Java_com_pfeprojet_carddetector_CardDetectorJNI_nativeDetectFromYUV(
    JNIEnv* env, jclass,
    jobject yBuffer, jobject uBuffer, jobject vBuffer,
    jint width, jint height,
    jint yRowStride, jint uvRowStride, jint uvPixelStride,
    jint rotationDegrees)
{
    jfloatArray jresult = env->NewFloatArray(RESULT_LEN);
    float data[RESULT_LEN] = {};

    if (!g_detector) {
        LOGE("not initialised");
        env->SetFloatArrayRegion(jresult, 0, RESULT_LEN, data);
        return jresult;
    }

    auto* yData = static_cast<uint8_t*>(env->GetDirectBufferAddress(yBuffer));
    if (!yData) {
        LOGE("null Y buffer");
        env->SetFloatArrayRegion(jresult, 0, RESULT_LEN, data);
        return jresult;
    }

    // ── Build grayscale Mat from Y plane (with stride handling) ──
    cv::Mat yMat(height, width, CV_8UC1);
    if (yRowStride == width) {
        memcpy(yMat.data, yData, width * height);
    } else {
        for (int r = 0; r < height; r++)
            memcpy(yMat.ptr(r), yData + r * yRowStride, width);
    }

    // ── Apply rotation to align raw sensor frame with screen orientation ──
    // Camera sensor is landscape; rotation corrects for phone portrait mode.
    cv::Mat grayRotated;
    switch (rotationDegrees) {
        case 90:
            cv::rotate(yMat, grayRotated, cv::ROTATE_90_CLOCKWISE);
            break;
        case 180:
            cv::rotate(yMat, grayRotated, cv::ROTATE_180);
            break;
        case 270:
            cv::rotate(yMat, grayRotated, cv::ROTATE_90_COUNTERCLOCKWISE);
            break;
        default:
            grayRotated = yMat;  // 0° — no copy
            break;
    }
    LOGI("nativeDetectFromYUV: %dx%d rot=%d \u2192 %dx%d",
         width, height, rotationDegrees, grayRotated.cols, grayRotated.rows);

    // ── Build Cr (V-plane) Mat for red validation ──
    // Android YUV_420_888: V plane = Cr channel, half resolution.
    cv::Mat crRotated;
    auto* vData = vBuffer ? static_cast<uint8_t*>(env->GetDirectBufferAddress(vBuffer)) : nullptr;
    if (vData) {
        int uvW = width  / 2;
        int uvH = height / 2;
        cv::Mat crMat(uvH, uvW, CV_8UC1);

        if (uvPixelStride == 1) {
            // Planar (I420/YV12)
            for (int r = 0; r < uvH; r++)
                memcpy(crMat.ptr(r), vData + r * uvRowStride, uvW);
        } else {
            // Semi-planar (NV21 / NV12): V bytes interleaved, stride >= uvW*2
            for (int r = 0; r < uvH; r++)
                for (int c = 0; c < uvW; c++)
                    crMat.at<uchar>(r, c) = vData[r * uvRowStride + c * uvPixelStride];
        }

        // Rotate Cr to match gray rotation
        switch (rotationDegrees) {
            case 90:  cv::rotate(crMat, crRotated, cv::ROTATE_90_CLOCKWISE);        break;
            case 180: cv::rotate(crMat, crRotated, cv::ROTATE_180);                 break;
            case 270: cv::rotate(crMat, crRotated, cv::ROTATE_90_COUNTERCLOCKWISE); break;
            default:  crRotated = crMat;                                             break;
        }
    }

    // ── Run detection (pass gray directly — no BGR conversion needed) ──
    g_detector->setCrMat(crRotated);
    auto result = g_detector->detectCard(grayRotated);
    fillResult(data, result);

    env->SetFloatArrayRegion(jresult, 0, RESULT_LEN, data);
    return jresult;
}

// ── Detect from Bitmap ──

JNIEXPORT jfloatArray JNICALL
Java_com_pfeprojet_carddetector_CardDetectorJNI_nativeDetectFromBitmap(
    JNIEnv* env, jclass, jobject bitmap)
{
    jfloatArray jresult = env->NewFloatArray(RESULT_LEN);
    float data[RESULT_LEN] = {};

    if (!g_detector) {
        env->SetFloatArrayRegion(jresult, 0, RESULT_LEN, data);
        return jresult;
    }

    AndroidBitmapInfo info;
    if (AndroidBitmap_getInfo(env, bitmap, &info) != ANDROID_BITMAP_RESULT_SUCCESS) {
        env->SetFloatArrayRegion(jresult, 0, RESULT_LEN, data);
        return jresult;
    }

    void* pixels = nullptr;
    AndroidBitmap_lockPixels(env, bitmap, &pixels);

    cv::Mat rgba(info.height, info.width, CV_8UC4, pixels);
    cv::Mat bgr;
    cv::cvtColor(rgba, bgr, cv::COLOR_RGBA2BGR);
    AndroidBitmap_unlockPixels(env, bitmap);

    auto result = g_detector->detectCard(bgr);
    fillResult(data, result);

    env->SetFloatArrayRegion(jresult, 0, RESULT_LEN, data);
    return jresult;
}

// ── Detect from grayscale byte[] ──

JNIEXPORT jfloatArray JNICALL
Java_com_pfeprojet_carddetector_CardDetectorJNI_nativeDetectFromGrayscale(
    JNIEnv* env, jclass,
    jbyteArray jdata, jint width, jint height)
{
    jfloatArray jresult = env->NewFloatArray(RESULT_LEN);
    float data[RESULT_LEN] = {};

    if (!g_detector) {
        env->SetFloatArrayRegion(jresult, 0, RESULT_LEN, data);
        return jresult;
    }

    jbyte* raw = env->GetByteArrayElements(jdata, nullptr);
    cv::Mat gray(height, width, CV_8UC1, reinterpret_cast<uint8_t*>(raw));
    cv::Mat bgr;
    cv::cvtColor(gray, bgr, cv::COLOR_GRAY2BGR);
    env->ReleaseByteArrayElements(jdata, raw, JNI_ABORT);

    auto result = g_detector->detectCard(bgr);
    fillResult(data, result);

    env->SetFloatArrayRegion(jresult, 0, RESULT_LEN, data);
    return jresult;
}

} // extern "C"
