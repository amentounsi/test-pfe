/**
 * Frame Processor Plugin Registration
 * Registers the detectCard frame processor with VisionCamera
 */

import { VisionCameraProxy, Frame } from 'react-native-vision-camera';
import type { CardDetectionResult } from '../types/cardDetection';

/**
 * Initialize the detectCard frame processor plugin
 */
const plugin = VisionCameraProxy.initFrameProcessorPlugin('detectCard', {});

/**
 * Detect card in a camera frame
 * This function is called by the frame processor worklet
 * 
 * @param frame - Camera frame from VisionCamera
 * @returns CardDetectionResult with detection status and corners
 */
export function detectCard(frame: Frame): CardDetectionResult {
  'worklet';
  
  if (plugin == null) {
    // Return error without console.warn (doesn't work in worklets)
    return {
      isValid: false,
      corners: [],
      error: 'Plugin not initialized',
    };
  }

  try {
    // Call native plugin
    const result = plugin.call(frame) as unknown as CardDetectionResult;
    
    // Validate result structure
    if (!result || typeof result !== 'object') {
      return {
        isValid: false,
        corners: [],
        error: 'Invalid result from plugin',
      };
    }

    // Ensure corners array has correct structure
    if (result.isValid && result.corners?.length === 4) {
      return {
        isValid: true,
        confidence: result.confidence,
        corners: result.corners as [
          { x: number; y: number },
          { x: number; y: number },
          { x: number; y: number },
          { x: number; y: number }
        ],
        frameWidth: result.frameWidth,
        frameHeight: result.frameHeight,
        orientation: result.orientation,
        debug: result.debug,
      };
    }

    return {
      isValid: false,
      corners: [],
      confidence: result.confidence,
      frameWidth: result.frameWidth,
      frameHeight: result.frameHeight,
      error: result.error,
      debug: result.debug,
    };
  } catch (error) {
    const errorMessage =
      error != null && typeof error === 'object' && 'message' in (error as Record<string, unknown>)
        ? String((error as Record<string, unknown>).message)
        : 'Native detectCard call failed';

    return {
      isValid: false,
      corners: [],
      error: errorMessage,
    };
  }
}

export default detectCard;
