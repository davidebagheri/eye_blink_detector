# eye_blink_detector
This package contains a modular pipeline of deep learning models to perform eye blink detection in real-time from a video stream. It is based on a SSD face detector, which detects all the faces in the current frame. Only one of the detected faces is analyzed computing the segmentations of both eyes with a UNet neural network, which will be the input to the final blink classifier, consisting of a LSTM model. 

<p align="center">
  <img src="imgs/gui.png" width="500" >
</p>


## Table of Contents
- [Installation](#Installation)
- [Example](#Example)


## Installation

## Example
