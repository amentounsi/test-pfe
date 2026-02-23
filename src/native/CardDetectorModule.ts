/**
 * Native Module Bridge for Card Detector
 * Provides TypeScript interface to native CardDetectorModule
 */

import { NativeModules, Platform } from 'react-native';
import type { CardDetectionConfig, CardDetectorConstants } from '../types/cardDetection';

const LINKING_ERROR =
  `The package 'CardDetectorModule' doesn't seem to be linked. Make sure: \n\n` +
  Platform.select({ ios: "- You have run 'pod install'\n", default: '' }) +
  '- You rebuilt the app after installing the package\n' +
  '- You are not using Expo Go\n';

/**
 * Native CardDetector module interface
 */
interface CardDetectorNativeModule {
  initialize(): Promise<boolean>;
  release(): Promise<boolean>;
  setConfig(
    cannyLow: number,
    cannyHigh: number,
    blurSize: number,
    minArea: number,
    maxArea: number,
    targetRatio: number,
    ratioTolerance: number
  ): Promise<boolean>;
  isInitialized(): Promise<boolean>;
  getConstants(): CardDetectorConstants;
}

/**
 * Get native module with error handling
 */
const CardDetectorNative: CardDetectorNativeModule = NativeModules.CardDetectorModule
  ? NativeModules.CardDetectorModule
  : new Proxy(
      {},
      {
        get() {
          throw new Error(LINKING_ERROR);
        },
      }
    );

/**
 * CardDetectorModule class
 * Wrapper for native module with TypeScript support
 */
class CardDetectorModule {
  private _isInitialized: boolean = false;

  /**
   * Initialize the card detector
   * Must be called before using detection
   */
  async initialize(): Promise<boolean> {
    if (this._isInitialized) {
      return true;
    }

    try {
      const result = await CardDetectorNative.initialize();
      this._isInitialized = result;
      return result;
    } catch (error) {
      console.error('Failed to initialize CardDetector:', error);
      throw error;
    }
  }

  /**
   * Release native resources
   */
  async release(): Promise<boolean> {
    if (!this._isInitialized) {
      return true;
    }

    try {
      const result = await CardDetectorNative.release();
      this._isInitialized = false;
      return result;
    } catch (error) {
      console.error('Failed to release CardDetector:', error);
      throw error;
    }
  }

  /**
   * Update detection configuration
   */
  async setConfig(config: Partial<CardDetectionConfig>): Promise<boolean> {
    const defaults = {
      cannyLowThreshold: 50,
      cannyHighThreshold: 150,
      blurKernelSize: 5,
      minAreaRatio: 0.02,
      maxAreaRatio: 0.85,
      targetAspectRatio: 1.586,
      aspectRatioTolerance: 0.35,
    };

    const mergedConfig = { ...defaults, ...config };

    try {
      return await CardDetectorNative.setConfig(
        mergedConfig.cannyLowThreshold,
        mergedConfig.cannyHighThreshold,
        mergedConfig.blurKernelSize,
        mergedConfig.minAreaRatio,
        mergedConfig.maxAreaRatio,
        mergedConfig.targetAspectRatio,
        mergedConfig.aspectRatioTolerance
      );
    } catch (error) {
      console.error('Failed to set CardDetector config:', error);
      throw error;
    }
  }

  /**
   * Check if detector is initialized
   */
  async isInitialized(): Promise<boolean> {
    try {
      return await CardDetectorNative.isInitialized();
    } catch {
      return false;
    }
  }

  /**
   * Get module constants
   */
  getConstants(): CardDetectorConstants {
    return CardDetectorNative.getConstants?.() ?? {
      ID1_ASPECT_RATIO: 1.586,
      DEFAULT_MIN_AREA: 0.005,
      DEFAULT_MAX_AREA: 0.95,
      DEFAULT_RATIO_TOLERANCE: 0.50,
    };
  }
}

// Export singleton instance
export const cardDetectorModule = new CardDetectorModule();
export default cardDetectorModule;
