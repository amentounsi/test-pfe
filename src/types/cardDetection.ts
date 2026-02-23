/**
 * Type definitions for Card Detection module
 * Tunisian ID Card (CIN) detection types
 */

export interface Point2D {
  x: number;
  y: number;
}

/** Debug info forwarded from native */
export interface DetectionDebugInfo {
  edgeWhitePixels: number;
  totalContours: number;
  topNContours: number;
  candidateQuads: number;
  bestScore: number;
  rejectedByArea: number;
  rejectedByApprox: number;
  rejectedByAspect: number;
  rejectedByEdgeDensity: number;
  largestAreaRatio: number;
}

/** Result of card detection from a single frame */
export interface CardDetectionResult {
  isValid: boolean;
  confidence?: number;
  corners: [Point2D, Point2D, Point2D, Point2D] | [];
  frameWidth?: number;
  frameHeight?: number;
  orientation?: string;
  error?: string;
  debug?: DetectionDebugInfo;
}

/** Configuration for card detection algorithm (mirrors DetectionConfig in C++) */
export interface CardDetectionConfig {
  cannyLowThreshold: number;
  cannyHighThreshold: number;
  blurKernelSize: number;
  minAreaRatio: number;
  maxAreaRatio: number;
  targetAspectRatio: number;
  aspectRatioTolerance: number;
}

/** Default configuration – matches CardDetector.h */
export const DEFAULT_DETECTION_CONFIG: CardDetectionConfig = {
  cannyLowThreshold: 50,
  cannyHighThreshold: 150,
  blurKernelSize: 5,
  minAreaRatio: 0.02,
  maxAreaRatio: 0.85,
  targetAspectRatio: 1.586,
  aspectRatioTolerance: 0.35,
};

/** Constants exposed by Turbo module */
export interface CardDetectorConstants {
  ID1_ASPECT_RATIO: number;
  DEFAULT_MIN_AREA: number;
  DEFAULT_MAX_AREA: number;
  DEFAULT_RATIO_TOLERANCE: number;
}

/** Overlay styles */
export interface OverlayStyle {
  validColor: string;
  invalidColor: string;
  strokeWidth: number;
  fillColor: string;
  cornerRadius?: number;
}
