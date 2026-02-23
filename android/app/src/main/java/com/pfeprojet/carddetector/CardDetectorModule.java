package com.pfeprojet.carddetector;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;

import com.facebook.react.bridge.ReactApplicationContext;
import com.facebook.react.bridge.ReactContextBaseJavaModule;
import com.facebook.react.bridge.ReactMethod;
import com.facebook.react.bridge.Promise;
import com.facebook.react.bridge.WritableMap;
import com.facebook.react.bridge.WritableArray;
import com.facebook.react.bridge.Arguments;
import com.facebook.react.module.annotations.ReactModule;

/**
 * React Native Module for Card Detection
 * Exposes card detection functionality to JavaScript
 */
@ReactModule(name = CardDetectorModule.NAME)
public class CardDetectorModule extends ReactContextBaseJavaModule {
    
    public static final String NAME = "CardDetectorModule";
    private boolean isInitialized = false;
    
    public CardDetectorModule(ReactApplicationContext reactContext) {
        super(reactContext);
    }
    
    @Override
    @NonNull
    public String getName() {
        return NAME;
    }
    
    /**
     * Initialize the card detector
     * Must be called before using detection
     */
    @ReactMethod
    public void initialize(Promise promise) {
        try {
            if (!isInitialized) {
                CardDetectorJNI.nativeInit();
                isInitialized = true;
            }
            promise.resolve(true);
        } catch (Exception e) {
            promise.reject("INIT_ERROR", "Failed to initialize CardDetector: " + e.getMessage());
        }
    }
    
    /**
     * Release native resources
     */
    @ReactMethod
    public void release(Promise promise) {
        try {
            if (isInitialized) {
                CardDetectorJNI.nativeRelease();
                isInitialized = false;
            }
            promise.resolve(true);
        } catch (Exception e) {
            promise.reject("RELEASE_ERROR", "Failed to release CardDetector: " + e.getMessage());
        }
    }
    
    /**
     * Update detection configuration
     */
    @ReactMethod
    public void setConfig(
        int cannyLow,
        int cannyHigh,
        int blurSize,
        double minArea,
        double maxArea,
        double targetRatio,
        double ratioTolerance,
        Promise promise
    ) {
        try {
            if (!isInitialized) {
                promise.reject("NOT_INITIALIZED", "CardDetector not initialized");
                return;
            }
            
            CardDetectorJNI.nativeSetConfig(
                cannyLow,
                cannyHigh,
                blurSize,
                (float) minArea,
                (float) maxArea,
                (float) targetRatio,
                (float) ratioTolerance
            );
            
            promise.resolve(true);
        } catch (Exception e) {
            promise.reject("CONFIG_ERROR", "Failed to set config: " + e.getMessage());
        }
    }
    
    /**
     * Check if detector is initialized
     */
    @ReactMethod
    public void isInitialized(Promise promise) {
        promise.resolve(isInitialized);
    }
    
    /**
     * Get module constants
     */
    @Nullable
    @Override
    public java.util.Map<String, Object> getConstants() {
        final java.util.Map<String, Object> constants = new java.util.HashMap<>();
        constants.put("ID1_ASPECT_RATIO", 1.586);
        constants.put("DEFAULT_MIN_AREA", 0.05);
        constants.put("DEFAULT_MAX_AREA", 0.85);
        constants.put("DEFAULT_RATIO_TOLERANCE", 0.10);
        return constants;
    }
}
