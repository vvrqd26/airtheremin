# 🎵 Air Theremin: Gesture-Controlled Synthesizer

## Project Overview

Air Theremin is an innovative music synthesizer that transforms hand gestures into sound using computer vision and signal processing. By leveraging MediaPipe hand tracking and real-time audio generation, users can create music through intuitive hand movements captured by a webcam.

## Key Features

### Version 1: Dual-Hand Control
- **Pitch Control**: Left hand's y-axis position determines musical note
- **Volume Control**: Right hand's thumb and index finger opening/closing controls volume
- Utilizes precise hand landmark tracking for nuanced musical expression

### Version 2: Simplified Interaction
- **Pitch Control**: Single hand's index finger y-axis position determines musical note
- **Volume Control**: Thumb and index finger opening/closing controls volume
- Reduced complexity for easier user interaction

### Version 3: Trump Accordion Edition
- **Unique Gesture Mapping**: Inspired by Trump's distinctive hand gestures
- **Volume Control**: Both hands' opening/closing distance determines volume
- **Pitch Control**: Fingertip positions on the y-axis determine musical note
- A playful and satirical take on gesture-based music generation

## Technical Stack
- Python
- OpenCV
- MediaPipe
- PyAudio
- NumPy
- Tkinter

## Requirements
- Python 3.8+
- OpenCV
- MediaPipe
- PyAudio
- NumPy

## Installation
```bash
pip install opencv-python mediapipe pyaudio numpy
```

## Usage
1. Connect a webcam
2. Run the desired version of the script
3. Move your hands to generate music
4. Press 'q' to quit the application

## How It Works
1. Captures webcam stream
2. Detects hand landmarks using MediaPipe
3. Translates hand positions into musical parameters
4. Generates real-time sine wave audio
5. Provides visual feedback on screen


## Contributing
Contributions, issues, and feature requests are welcome! Feel free to check the issues page.



## Author
[LuYa]

## Acknowledgments
- MediaPipe for hand tracking
- OpenCV community
- Python audio libraries

