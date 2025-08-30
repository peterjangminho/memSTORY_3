#!/bin/bash

# KV Cache Fix Testing Script
# Tests the improved KV cache implementation with ONNX Runtime 1.23.0

echo "🚀 memSTORY KV Cache Optimization Test Script"
echo "=============================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    case $2 in
        "ERROR") echo -e "${RED}❌ $1${NC}" ;;
        "SUCCESS") echo -e "${GREEN}✅ $1${NC}" ;;
        "WARNING") echo -e "${YELLOW}⚠️ $1${NC}" ;;
        "INFO") echo -e "${BLUE}ℹ️ $1${NC}" ;;
        *) echo "$1" ;;
    esac
}

# Check Android SDK and environment
print_status "Checking Android development environment..." "INFO"
if [ -z "$ANDROID_HOME" ]; then
    print_status "ANDROID_HOME not set. Please set Android SDK path." "ERROR"
    exit 1
fi

# Clean and build
print_status "Cleaning previous build..." "INFO"
./gradlew clean

print_status "Building with ONNX Runtime 1.23.0..." "INFO"
./gradlew assembleDebug

if [ $? -eq 0 ]; then
    print_status "Build successful!" "SUCCESS"
else
    print_status "Build failed. Check dependencies and ONNX Runtime version." "ERROR"
    exit 1
fi

# Install APK if device connected
if adb devices | grep -q "device"; then
    print_status "Installing APK on connected device..." "INFO"
    adb install -r app/build/outputs/apk/debug/app-debug.apk
    
    if [ $? -eq 0 ]; then
        print_status "APK installed successfully!" "SUCCESS"
        
        print_status "Starting app to test KV cache..." "INFO"
        adb shell am start -n com.memstory/.presentation.MainActivity
        
        print_status "Monitoring logs for KV cache updates..." "INFO"
        echo "📱 Watch for these log messages:"
        echo "   - 'KV cache updated successfully (direct)'"
        echo "   - 'KV cache updated successfully (converted)'"
        echo "   - 'Failed to update KV cache' should be REDUCED/ELIMINATED"
        echo ""
        echo "🔍 Use this command to monitor logs:"
        echo "   adb logcat | grep -E '(OnnxLLMEngine|KV cache)'"
        
    else
        print_status "APK installation failed" "ERROR"
    fi
else
    print_status "No Android device connected. APK build completed." "WARNING"
    print_status "Connect device and run: adb install -r app/build/outputs/apk/debug/app-debug.apk" "INFO"
fi

echo ""
print_status "KV Cache Optimization Applied:" "SUCCESS"
echo "  • ONNX Runtime upgraded: 1.22.0 → 1.23.0"
echo "  • Added safe tensor casting with convertArrayToTensor()"
echo "  • Enhanced error handling for Java array ↔ OnnxTensor conversion"
echo "  • Improved logging for KV cache update debugging"
echo ""
print_status "Test the app and monitor logs for KV cache improvements!" "INFO"