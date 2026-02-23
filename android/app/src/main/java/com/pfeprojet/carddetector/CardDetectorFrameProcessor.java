package com.pfeprojet.carddetector;

import android.media.Image;
import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.camera.core.ImageProxy;

import com.mrousavy.camera.frameprocessor.Frame;
import com.mrousavy.camera.frameprocessor.FrameProcessorPlugin;
import com.mrousavy.camera.frameprocessor.VisionCameraProxy;

import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * VisionCamera Frame Processor Plugin for Card Detection
 * Processes camera frames in real-time to detect ID cards
 */
public class CardDetectorFrameProcessor extends FrameProcessorPlugin {
    
    private static final String TAG = "CardDetectorFrameProcessor";
    private boolean isInitialized = false;
    
    public CardDetectorFrameProcessor(@NonNull VisionCameraProxy proxy, @Nullable Map<String, Object> options) {
        super();
        // Initialize native detector
        try {
            CardDetectorJNI.nativeInit();
            isInitialized = true;
        } catch (Exception e) {
            android.util.Log.e(TAG, "Failed to initialize CardDetector: " + e.getMessage());
        }
    }
    
    private static int frameCount = 0;
    
    @Nullable
    @Override
    public Object callback(@NonNull Frame frame, @Nullable Map<String, Object> arguments) {
        frameCount++;
        
        if (frameCount % 60 == 1) {
            android.util.Log.d(TAG, "callback() called - frame #" + frameCount + ", isInitialized=" + isInitialized);
        }
        
        if (!isInitialized) {
            android.util.Log.e(TAG, "Detector not initialized!");
            return createErrorResult("Detector not initialized");
        }
        
        try {
            // Get image from frame
            Image image = frame.getImage();
            if (image == null) {
                android.util.Log.e(TAG, "No image in frame");
                return createErrorResult("No image in frame");
            }
            
            int width = image.getWidth();
            int height = image.getHeight();
            
            if (frameCount % 60 == 1) {
                android.util.Log.d(TAG, "Frame size: " + width + "x" + height);
            }
            
            // Get Y plane (luminance) for grayscale processing
            Image.Plane[] planes = image.getPlanes();
            if (planes.length == 0) {
                android.util.Log.e(TAG, "No image planes");
                return createErrorResult("No image planes");
            }
            
            Image.Plane yPlane = planes[0];
            ByteBuffer yBuffer = yPlane.getBuffer();
            int yRowStride = yPlane.getRowStride();

            ByteBuffer uBuffer = planes.length > 1 ? planes[1].getBuffer() : null;
            ByteBuffer vBuffer = planes.length > 2 ? planes[2].getBuffer() : null;
            int uvRowStride   = planes.length > 1 ? planes[1].getRowStride()   : 0;
            int uvPixelStride = planes.length > 1 ? planes[1].getPixelStride() : 0;

            // Map VisionCamera orientation string to rotation degrees.
            // The raw sensor frame is landscape; rotation aligns it with screen.
            String orientStr = "portrait-up";
            try { orientStr = frame.getOrientation().toString(); } catch (Throwable ignored) {}
            int rotationDegrees;
            switch (orientStr.toLowerCase()) {
                case "portrait":        // VisionCamera 3.x enum
                case "portrait_up":
                case "portrait-up":     rotationDegrees = 90;  break;
                case "portrait_down":
                case "portrait-down":   rotationDegrees = 270; break;
                case "landscape_left":
                case "landscape-left":  rotationDegrees = 180; break;
                default:                rotationDegrees = 0;   break; // landscape-right or unknown
            }
            if (frameCount % 60 == 1)
                android.util.Log.d(TAG, "Frame orientation: " + orientStr + " → rot=" + rotationDegrees + "°");

            // Detect card using native code
            float[] result = CardDetectorJNI.nativeDetectFromYUV(
                yBuffer, uBuffer, vBuffer,
                width, height,
                yRowStride, uvRowStride, uvPixelStride,
                rotationDegrees
            );

            // Use rotated dimensions for the response
            int outW = (rotationDegrees == 90 || rotationDegrees == 270) ? height : width;
            int outH = (rotationDegrees == 90 || rotationDegrees == 270) ? width  : height;
            
            if (frameCount % 60 == 1) {
                android.util.Log.d(TAG, "Native result: " + (result != null ? "length=" + result.length + ", isValid=" + (result.length > 0 ? result[0] : "null") : "null"));
            }

            // Parse result using rotated dimensions
            return parseDetectionResult(result, outW, outH, orientStr);
            
        } catch (Throwable e) {
            android.util.Log.e(TAG, "Error processing frame: " + e.getMessage(), e);
            return createErrorResult(e.getMessage());
        }
    }
    
    /**
     * Parse native detection result into JavaScript-friendly format.
     * Native returns float[20]:
     *   [0] isValid, [1] confidence, [2..9] corners,
     *   [10] edgeWhitePixels, [11] totalContours, [12] candidateQuads,
     *   [13] bestScore, [14] topNContours,
     *   [15] rejectedByArea, [16] rejectedByApprox, [17] rejectedByAspect,
     *   [18] largestContourAreaRatio, [19] rejectedByEdgeDensity
     *
     * IMPORTANT: VisionCamera only supports Boolean, Integer, Double (NOT Float), String, Map, List
     */
    private Map<String, Object> parseDetectionResult(float[] result, int width, int height, String orientation) {
        Map<String, Object> response = new HashMap<>();
        
        if (result == null || result.length < 20) {
            response.put("isValid", Boolean.FALSE);
            response.put("corners", new ArrayList<>());
            response.put("frameWidth", Integer.valueOf(width));
            response.put("frameHeight", Integer.valueOf(height));
            return response;
        }
        
        boolean isValid = result[0] > 0.5f;
        double confidence = (double) result[1];
        
        response.put("isValid", Boolean.valueOf(isValid));
        response.put("confidence", Double.valueOf(confidence));
        response.put("frameWidth", Integer.valueOf(width));
        response.put("frameHeight", Integer.valueOf(height));
        response.put("orientation", orientation);
        
        if (isValid) {
            List<Map<String, Object>> corners = new ArrayList<>();
            for (int i = 0; i < 4; i++) {
                Map<String, Object> corner = new HashMap<>();
                corner.put("x", Double.valueOf((double) result[2 + i * 2]));
                corner.put("y", Double.valueOf((double) result[3 + i * 2]));
                corners.add(corner);
            }
            response.put("corners", corners);
        } else {
            response.put("corners", new ArrayList<>());
        }
        
        // Debug fields (always sent so overlay can display them)
        Map<String, Object> debug = new HashMap<>();
        debug.put("edgeWhitePixels", Integer.valueOf((int) result[10]));
        debug.put("totalContours",   Integer.valueOf((int) result[11]));
        debug.put("candidateQuads",  Integer.valueOf((int) result[12]));
        debug.put("bestScore",       Double.valueOf((double) result[13]));
        debug.put("topNContours",    Integer.valueOf((int) result[14]));
        debug.put("rejectedByArea",  Integer.valueOf((int) result[15]));
        debug.put("rejectedByApprox", Integer.valueOf((int) result[16]));
        debug.put("rejectedByAspect", Integer.valueOf((int) result[17]));
        debug.put("largestAreaRatio", Double.valueOf((double) result[18]));
        debug.put("rejectedByEdgeDensity", Integer.valueOf((int) result[19]));
        response.put("debug", debug);
        
        return response;
    }
    
    /**
     * Create error result
     */
    private Map<String, Object> createErrorResult(String message) {
        Map<String, Object> response = new HashMap<>();
        response.put("isValid", Boolean.FALSE);
        response.put("error", message);
        response.put("corners", new ArrayList<>());
        return response;
    }
}
