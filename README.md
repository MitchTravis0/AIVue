# AIVue - Hands-free Computer Control System

AIVue is an accessibility solution designed for individuals with physical disabilities that limit their motor functions, enabling hands-free computer control through head movements, eye gestures, and voice commands.

## Overview

This system combines multiple input methods to provide a comprehensive hands-free computing experience:

1. **Head Gaze Tracking** - Control your cursor by moving your head
2. **Wink/Blink Detection** - Execute clicks through eye gestures
3. **Speech-to-Text** - Type and input text using your voice
4. **Computer Use Agent** - Execute complex tasks through natural language commands using Google's Gemini AI

## Components

- **pipefacemac.py**: Head tracking and eye gesture system using MediaPipe and computer vision
- **STT.py**: Speech-to-text functionality using Vosk for offline speech recognition
- **action_executor_mac.py**: Natural language command interpreter using Google's Gemini API
- **executions.py**: System action execution engine (keyboard, mouse control, etc.)
- **pygui.py**: Simple GUI launcher for the system

## Features

- **Head-controlled Mouse**: Move the cursor by moving your head
- **Eye Gesture Detection**: 
  - Left wink/blink for left mouse click
  - Right wink/blink for right mouse click
  - Double wink detection for double-click
- **Voice-to-Text**: Type by speaking
- **AI Task Execution**: Natural language commands to perform complex tasks
- **Smart Filtering**: Kalman filtering for smooth cursor movement
- **Calibration System**: Personalized calibration for accurate head tracking

## Requirements

- Python 3.8 or higher
- Webcam 
- Microphone
- macOS (primary support) / Windows / Linux (with some limitations)
- Google Gemini API key (for the brain of the CUA)

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/MitchTravis0/AIVue.git
   cd AIVue
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Download Vosk speech model:
   Download the model from https://alphacephei.com/vosk/models and extract it to the project directory.

4. Configure your API key:
   Create a `.env` file with your Google Gemini API key:
   ```
   GEMINI_KEY=your_api_key_here
   ```

## Usage

### Starting the Application

Launch the application through the GUI:
```
python pygui.py
```

This will open a launcher interface with options to start calibration.

### Calibration

1. Follow the on-screen instructions to calibrate head tracking
2. Look at each calibration point as directed

### Using Head Tracking

- Move your head to control the cursor position
- Wink/blink with left eye for left click
- Wink/blink with right eye for right click
- Double wink for double-click

### Using Voice Input

- Speech is automatically transcribed to text
- Speak clearly toward your microphone

### Using AI Command Execution

The system can interpret natural language commands like:
- "Open Google Chrome"
- "Search for Python documentation"

## Troubleshooting

- **Camera not found**: Ensure your camera is connected and permissions are granted
- **Speech recognition issues**: Check your microphone settings and ensure the Vosk model is correctly installed
- **Head tracking not working**: Run calibration again in good lighting conditions


## Acknowledgments

- MediaPipe for face mesh tracking
- Vosk for speech recognition
- Google Generative AI for natural language processing
- filterpy for Kalman filtering 