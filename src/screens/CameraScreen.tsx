/**
 * CameraScreen Component
 * Main camera screen with real-time card detection overlay
 */

import React, { useCallback, useEffect, useState, useRef, useMemo } from 'react';
import {
  StyleSheet,
  View,
  Text,
  Dimensions,
  StatusBar,
  Platform,
  PermissionsAndroid,
  Alert,
  ActivityIndicator,
} from 'react-native';
import {
  Camera,
  useCameraDevice,
  useCameraPermission,
  CameraPosition,
} from 'react-native-vision-camera';
import { useCardDetection } from '../hooks/useCardDetection';
import CardOverlay, { CardGuideFrame, calculateOverlayBounds } from '../components/CardOverlay';
import type { CardDetectionResult } from '../types/cardDetection';

/**
 * Screen dimensions
 */
const { width: SCREEN_WIDTH, height: SCREEN_HEIGHT } = Dimensions.get('window');

/**
 * Camera screen props
 */
interface CameraScreenProps {
  /** Camera position */
  cameraPosition?: CameraPosition;
  
  /** Enable torch */
  enableTorch?: boolean;
  
  /** Callback when card is detected */
  onCardDetected?: (result: CardDetectionResult) => void;
  
  /** Show debug info */
  showDebugInfo?: boolean;
}

/**
 * CameraScreen component
 */
export const CameraScreen: React.FC<CameraScreenProps> = ({
  cameraPosition = 'back',
  enableTorch = false,
  onCardDetected,
  showDebugInfo = true,
}) => {
  // Camera permission
  const { hasPermission, requestPermission } = useCameraPermission();
  
  // Camera device
  const device = useCameraDevice(cameraPosition);
  
  // State
  const [isActive, setIsActive] = useState(true);
  const [viewDimensions, setViewDimensions] = useState({
    width: SCREEN_WIDTH,
    height: SCREEN_HEIGHT,
  });
  
  // Camera ref
  const cameraRef = useRef<Camera>(null);
  
  // State for overlay bounds and configuration
  const [overlayEnabled, setOverlayEnabled] = useState(false);
  const [overlayBounds, setOverlayBounds] = useState<{
    x: number;
    y: number;
    width: number;
    height: number;
  } | null>(null);
  
  // Card detection hook
  const {
    detectionResult,
    isReady,
    frameProcessor,
    scaledCorners,
  } = useCardDetection({
    enabled: isActive && hasPermission,
    onCardDetected,
    throttleMs: 50, // Update every 50ms for smoother overlay
    useOverlay: overlayEnabled,
    overlayBounds,
    useROICropping: false,  // Full frame detection; overlay used for constraint validation only
  });
  
  // Calculate and set overlay bounds once we have frame dimensions
  useEffect(() => {
    if (detectionResult?.frameWidth && detectionResult?.frameHeight && !overlayEnabled) {
      const bounds = calculateOverlayBounds(
        detectionResult.frameWidth,
        detectionResult.frameHeight,
        viewDimensions.width,
        viewDimensions.height,
        1.586, // CIN aspect ratio
        40     // padding
      );
      setOverlayBounds(bounds);
      setOverlayEnabled(true);
      console.log('Overlay bounds calculated:', bounds);
    }
  }, [detectionResult?.frameWidth, detectionResult?.frameHeight, viewDimensions, overlayEnabled]);

  /**
   * Request camera permission on mount
   */
  useEffect(() => {
    const requestCameraPermission = async () => {
      if (Platform.OS === 'android') {
        try {
          const granted = await PermissionsAndroid.request(
            PermissionsAndroid.PERMISSIONS.CAMERA,
            {
              title: 'Camera Permission',
              message: 'This app needs access to your camera to scan ID cards.',
              buttonNeutral: 'Ask Me Later',
              buttonNegative: 'Cancel',
              buttonPositive: 'OK',
            }
          );
          
          if (granted !== PermissionsAndroid.RESULTS.GRANTED) {
            Alert.alert(
              'Permission Required',
              'Camera permission is required to scan ID cards.'
            );
          }
        } catch (err) {
          console.error('Error requesting camera permission:', err);
        }
      } else {
        await requestPermission();
      }
    };

    if (!hasPermission) {
      requestCameraPermission();
    }
  }, [hasPermission, requestPermission]);

  /**
   * Handle view layout to get dimensions
   */
  const onLayout = useCallback((event: any) => {
    const { width, height } = event.nativeEvent.layout;
    setViewDimensions({ width, height });
  }, []);

  /**
   * Render loading state
   */
  if (!hasPermission) {
    return (
      <View style={styles.container}>
        <StatusBar barStyle="light-content" backgroundColor="#000" />
        <View style={styles.loadingContainer}>
          <ActivityIndicator size="large" color="#00FF00" />
          <Text style={styles.loadingText}>Requesting camera permission...</Text>
        </View>
      </View>
    );
  }

  /**
   * Render no device state
   */
  if (device == null) {
    return (
      <View style={styles.container}>
        <StatusBar barStyle="light-content" backgroundColor="#000" />
        <View style={styles.errorContainer}>
          <Text style={styles.errorText}>No camera device found</Text>
        </View>
      </View>
    );
  }

  /**
   * Render camera not ready state
   */
  if (!isReady) {
    return (
      <View style={styles.container}>
        <StatusBar barStyle="light-content" backgroundColor="#000" />
        <View style={styles.loadingContainer}>
          <ActivityIndicator size="large" color="#00FF00" />
          <Text style={styles.loadingText}>Initializing card detector...</Text>
        </View>
      </View>
    );
  }

  return (
    <View style={styles.container} onLayout={onLayout}>
      <StatusBar barStyle="light-content" backgroundColor="#000" translucent />
      
      {/* Camera */}
      <Camera
        ref={cameraRef}
        style={StyleSheet.absoluteFill}
        device={device}
        isActive={isActive}
        frameProcessor={frameProcessor}
        torch={enableTorch ? 'on' : 'off'}
        pixelFormat="yuv"
        orientation="portrait"
      />
      
      {/* Fixed Guide Frame Overlay */}
      <CardGuideFrame
        viewWidth={viewDimensions.width}
        viewHeight={viewDimensions.height}
        aspectRatio={1.586}
        padding={40}
        showValidation={(detectionResult?.debug?.temporalValidCount ?? 0) > 0 || detectionResult?.isValid === true}
        isAligned={detectionResult?.isValid || false}
      />
      
      {/* Detection Overlay - Shows detected card corners when valid */}
      {detectionResult?.isValid && detectionResult.corners.length === 4 && (
        <CardOverlay
          corners={detectionResult.corners}
          frameWidth={detectionResult.frameWidth || viewDimensions.width}
          frameHeight={detectionResult.frameHeight || viewDimensions.height}
          viewWidth={viewDimensions.width}
          viewHeight={viewDimensions.height}
          isValid={detectionResult.isValid}
          showCornerMarkers={true}
          showEdgeLines={true}
        />
      )}
      
      {/* Instructions removed - now shown by CardGuideFrame */}
      <View style={styles.instructionsContainer}>
        <Text style={styles.instructionsText}>
          Position your ID card within the camera view
        </Text>
        {detectionResult?.isValid && (
          <Text style={styles.detectedText}>Card Detected!</Text>
        )}
      </View>
      
      {/* Debug Info – stage-by-stage visibility */}
      {showDebugInfo && (
        <View style={styles.debugContainer}>
          <Text style={styles.debugText}>
            Detection: {detectionResult?.isValid ? 'VALID' : 'NONE'}
          </Text>
          <Text style={styles.debugText}>
            Confidence: {detectionResult?.confidence?.toFixed(3) || 'N/A'}
          </Text>
          <Text style={styles.debugText}>
            Frame: {detectionResult?.frameWidth}x{detectionResult?.frameHeight}
          </Text>
          <Text style={styles.debugText}>
            View: {viewDimensions.width.toFixed(0)}x{viewDimensions.height.toFixed(0)}
          </Text>
          {detectionResult?.debug && (
            <>
              <Text style={styles.debugSeparator}>── Pipeline ──</Text>
              <Text style={styles.debugText}>
                S1 edges: {detectionResult.debug.edgeWhitePixels} px
              </Text>
              <Text style={styles.debugText}>
                S2 contours: {detectionResult.debug.totalContours} → top {detectionResult.debug.topNContours}
              </Text>
              <Text style={styles.debugText}>
                S2 largest: {((detectionResult.debug.largestAreaRatio || 0) * 100).toFixed(2)}%
              </Text>
              <Text style={styles.debugSeparator}>── Stage 3 ──</Text>
              <Text style={styles.debugText}>
                S3 quads: {detectionResult.debug.candidateQuads}
              </Text>
              <Text style={styles.debugText}>
                rej area: {detectionResult.debug.rejectedByArea ?? '?'} | approx: {detectionResult.debug.rejectedByApprox ?? '?'} | aspect: {detectionResult.debug.rejectedByAspect ?? '?'} | edge: {detectionResult.debug.rejectedByEdgeDensity ?? '?'}
              </Text>
              <Text style={styles.debugText}>
                S4 best: {detectionResult.debug.bestScore?.toFixed(3) || '—'}
              </Text>
            </>
          )}
        </View>
      )}
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#000',
  },
  loadingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#000',
  },
  loadingText: {
    color: '#fff',
    fontSize: 16,
    marginTop: 16,
  },
  errorContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#000',
  },
  errorText: {
    color: '#ff0000',
    fontSize: 18,
    textAlign: 'center',
  },
  instructionsContainer: {
    position: 'absolute',
    top: 60,
    left: 0,
    right: 0,
    alignItems: 'center',
    paddingHorizontal: 20,
  },
  instructionsText: {
    color: '#fff',
    fontSize: 16,
    textAlign: 'center',
    backgroundColor: 'rgba(0, 0, 0, 0.5)',
    paddingHorizontal: 16,
    paddingVertical: 8,
    borderRadius: 8,
  },
  detectedText: {
    color: '#00FF00',
    fontSize: 20,
    fontWeight: 'bold',
    marginTop: 16,
    backgroundColor: 'rgba(0, 0, 0, 0.5)',
    paddingHorizontal: 16,
    paddingVertical: 8,
    borderRadius: 8,
  },
  debugContainer: {
    position: 'absolute',
    bottom: 40,
    left: 16,
    backgroundColor: 'rgba(0, 0, 0, 0.7)',
    padding: 12,
    borderRadius: 8,
  },
  debugText: {
    color: '#00FF00',
    fontSize: 12,
    fontFamily: Platform.OS === 'android' ? 'monospace' : 'Courier',
    marginBottom: 4,
  },
  debugSeparator: {
    color: '#888',
    fontSize: 10,
    fontFamily: Platform.OS === 'android' ? 'monospace' : 'Courier',
    marginTop: 4,
    marginBottom: 2,
  },
});

export default CameraScreen;
