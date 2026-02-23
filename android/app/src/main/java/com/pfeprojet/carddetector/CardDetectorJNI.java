package com.pfeprojet.carddetector;

import android.graphics.Bitmap;
import java.nio.ByteBuffer;

/**
 * JNI wrapper for native CardDetector functions
 * Provides interface between Java and C++ OpenCV code
 */
public class CardDetectorJNI {
    
    static {
        System.loadLibrary("carddetector");
    }
    
    /**
     * Initialize the native CardDetector
     * Must be called before any detection
     */
    public static native void nativeInit();
    
    /**
     * Release native resources
     * Call when done with detection
     */
    public static native void nativeRelease();
    
    /**
     * Update detection configuration
     * 
     * @param cannyLow Canny edge detection low threshold
     * @param cannyHigh Canny edge detection high threshold
     * @param blurSize Gaussian blur kernel size
     * @param minArea Minimum area ratio (0-1)
     * @param maxArea Maximum area ratio (0-1)
     * @param targetRatio Target aspect ratio (ID-1 = 1.586)
     * @param ratioTolerance Aspect ratio tolerance (0-1)
     */
    public static native void nativeSetConfig(
        int cannyLow,
        int cannyHigh,
        int blurSize,
        float minArea,
        float maxArea,
        float targetRatio,
        float ratioTolerance
    );
    
    /**
     * Detect card from YUV frame data
     * Optimized for camera frames
     *
     * @param yBuffer        Y plane direct buffer
     * @param uBuffer        U plane direct buffer
     * @param vBuffer        V plane direct buffer (Cr — used for red validation)
     * @param width          Frame width
     * @param height         Frame height
     * @param yRowStride     Y plane row stride
     * @param uvRowStride    UV plane row stride
     * @param uvPixelStride  UV pixel stride (1=planar, 2=semi-planar)
     * @param rotationDegrees Rotation to apply: 0, 90, 180, 270
     * @return float[20]
     */
    public static native float[] nativeDetectFromYUV(
        ByteBuffer yBuffer,
        ByteBuffer uBuffer,
        ByteBuffer vBuffer,
        int width,
        int height,
        int yRowStride,
        int uvRowStride,
        int uvPixelStride,
        int rotationDegrees
    );
    
    /**
     * Detect card from RGBA bitmap
     * 
     * @param bitmap Android Bitmap (ARGB_8888)
     * @return float[10]: [isValid, confidence, x0, y0, x1, y1, x2, y2, x3, y3]
     */
    public static native float[] nativeDetectFromBitmap(Bitmap bitmap);
    
    /**
     * Detect card from grayscale byte array
     * Most efficient method for processed frames
     * 
     * @param data Grayscale image data
     * @param width Image width
     * @param height Image height
     * @return float[10]: [isValid, confidence, x0, y0, x1, y1, x2, y2, x3, y3]
     */
    public static native float[] nativeDetectFromGrayscale(
        byte[] data,
        int width,
        int height
    );
}
