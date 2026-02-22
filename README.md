# Tunisian ID Card (CIN) Detection Module

Real-time detection of Tunisian ID cards using React Native Vision Camera, OpenCV, and dynamic SVG overlay.

## Features

- ✅ Real-time card detection at 30-60 FPS
- ✅ OpenCV-based edge detection and contour analysis
- ✅ Geometric validation (convexity, aspect ratio, area)
- ✅ Dynamic SVG overlay with corner markers
- ✅ 100% offline processing
- ✅ TypeScript support

## Requirements

- Node.js 18 LTS
- Java 17
- Android SDK API 34
- NDK 25.2
- CMake 3.22+
- OpenCV Android SDK 4.x

## Installation

### 1. Install Node dependencies

```bash
npm install
```

### 2. Download OpenCV Android SDK

Download OpenCV Android SDK from [opencv.org](https://opencv.org/releases/) and extract it to:

```
android/app/src/main/cpp/opencv/
```

The structure should be:
```
android/app/src/main/cpp/
├── opencv/
│   └── sdk/
│       └── native/
│           ├── jni/
│           └── libs/
├── CardDetector.cpp
├── CardDetector.h
├── CardDetectorJNI.cpp
└── CMakeLists.txt
```

### 3. Configure local.properties

Create `android/local.properties`:

```properties
sdk.dir=C:\\Users\\USER\\AppData\\Local\\Android\\Sdk
ndk.dir=C:\\Users\\USER\\AppData\\Local\\Android\\Sdk\\ndk\\25.2.9519653
cmake.dir=C:\\Users\\USER\\AppData\\Local\\Android\\Sdk\\cmake\\3.22.1
```

### 4. Build and Run

```bash
# Clean build
npm run clean:android

# Run on device
npm run android
```

## Architecture

```
├── android/
│   └── app/src/main/
│       ├── cpp/                          # Native C++ code
│       │   ├── CardDetector.cpp          # OpenCV detection logic
│       │   ├── CardDetector.h            # Header file
│       │   ├── CardDetectorJNI.cpp       # JNI bridge
│       │   └── CMakeLists.txt            # CMake config
│       └── java/com/pfeprojet/
│           └── carddetector/
│               ├── CardDetectorJNI.java  # JNI wrapper
│               ├── CardDetectorModule.java
│               ├── CardDetectorPackage.java
│               ├── CardDetectorFrameProcessor.java
│               └── CardDetectorPluginProvider.java
├── src/
│   ├── components/
│   │   └── CardOverlay.tsx               # SVG overlay
│   ├── frameProcessor/
│   │   └── detectCard.ts                 # Frame processor plugin
│   ├── hooks/
│   │   └── useCardDetection.ts           # Detection hook
│   ├── native/
│   │   └── CardDetectorModule.ts         # Native module bridge
│   ├── screens/
│   │   └── CameraScreen.tsx              # Main camera screen
│   └── types/
│       └── cardDetection.ts              # TypeScript types
└── App.tsx                               # Entry point
```

## Detection Algorithm

### 1. Preprocessing
- Convert to grayscale
- Apply Gaussian Blur (5x5 kernel)
- Apply Canny Edge Detection

### 2. Contour Detection
- Find contours with `findContours()`
- Approximate polygons with `approxPolyDP()`
- Filter for 4-vertex convex polygons

### 3. Geometric Validation
- **Area**: 20% - 85% of image area
- **Aspect Ratio**: ID-1 standard (1.586 ± 10%)
- **Convexity**: Must be convex
- **Bounds**: All points within image

### 4. Corner Sorting
Corners are sorted: top-left → top-right → bottom-right → bottom-left

## API

### useCardDetection Hook

```typescript
const {
  detectionResult,  // Current detection result
  isReady,          // Whether detector is initialized
  frameProcessor,   // Frame processor for VisionCamera
  scaledCorners,    // Scaled corners for overlay
  reset,            // Reset detection state
  updateConfig,     // Update detection config
} = useCardDetection({
  enabled: true,
  onCardDetected: (result) => console.log(result),
  throttleMs: 100,
});
```

### CardDetectionResult

```typescript
interface CardDetectionResult {
  isValid: boolean;
  confidence?: number;
  corners: [Point2D, Point2D, Point2D, Point2D] | [];
  frameWidth?: number;
  frameHeight?: number;
}
```

## Configuration

```typescript
const config: CardDetectionConfig = {
  cannyLowThreshold: 50,
  cannyHighThreshold: 150,
  blurKernelSize: 5,
  minAreaRatio: 0.20,
  maxAreaRatio: 0.85,
  targetAspectRatio: 1.586,
  aspectRatioTolerance: 0.10,
};
```

## Performance

- Target: < 50ms per frame
- No memory allocation per frame
- All heavy processing in C++
- Reusable Mat objects
- Efficient YUV to grayscale conversion

## Troubleshooting

### Build Errors

1. **OpenCV not found**: Ensure OpenCV SDK is in the correct path
2. **NDK errors**: Verify NDK version 25.2 is installed
3. **CMake errors**: Check CMake 3.22+ is available

### Runtime Issues

1. **Camera permission**: Ensure CAMERA permission is granted
2. **No detection**: Check lighting conditions
3. **Low FPS**: Reduce processing resolution

## License

MIT License - Open source only
