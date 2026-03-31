# Emotion-Sentiment-Analyzer
Realtime emotion detection using face and voice analysis. A multimodal approach to understanding human behavior.An intelligent system that understands emotions through face and voice. Bringing humanlike perception to machines using AI techniques.

## Overview

This project is a real time Emotion & Sentiment Analyzer that uses facial expressions and voice input to determine a user’s emotional state.
It combines computer vision and NLP techniques into a single interactive application.
This project simulates an intelligent humanaware system capable of understanding emotions through both facial expressions and voice. By combining visual perception and language analysis, it attempts to bridge the gap between human behavior and machine understanding.
The application operates in real time, capturing facial data via webcam and analyzing speech sentiment through audio input. A fusion mechanism integrates both signals to provide a more accurate emotional interpretation.
With gesturebased interaction and a stable output system, the project demonstrates how multimodal AI can enhance user experience in future humancomputer interaction systems.


## Features

*  Realtime face detection
*  Voice sentiment analysis
*  Gesturebased controls
*  Freeze mode for stable output
*  Emotion history tracking
*  Interactive GUI using Tkinter



## Tech Stack

* Python
* OpenCV
* Tkinter
* SpeechRecognition
* TextBlob
* NumPy



## Installation


pip install opencvpython pillow numpy textblob speechrecognition sounddevice

*(Satisfy these dependencies too if not preinstalled)Built-in modules like:

tkinter
threading
queue
wave
time




## How to Run


python main.py




## Controls

| Action        | Method                |
|
| Record Audio  | Button / Palm Gesture |
| Upload Image  | Button                |
| Capture Image | Hold Gesture          |
| Freeze Output | Button                |
When an image is uploaded, the system automatically processes it in the background without requiring any extra steps. The detected emotion is then added to the result history, allowing users to track recent outputs easily.

This makes the interaction smoother and avoids unnecessary manual actions.


## Project Structure

```
main.py
README.md
Requirments.txt
Statment.md

```



## Limitations

* Uses heuristic emotion detection
* Detection depends on lighting
* Requires internet for speech recognition



## Future Scope

* Deep learningbased emotion model
* MediaPipe gesture tracking
* Higher accuracy and robustness



## Author

Uday Deval Chaturvedi
Regno. 25BCE10300
VIT BHOPAL UNIVERSITY



## License

This project is for academic and educational purposes.


