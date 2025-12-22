===========================================
 Ethnicity Classification - FYP Prototype
===========================================

Description:
-------------
This is a Final Year Project (FYP) prototype that uses deep learning
to classify Malaysian ethnicity (Malay, Chinese, Indian) through
a webcam in real-time.

The app detects faces using OpenCV and predicts ethnicity using the 
trained model.

How to Run:
-------------
1. Make sure your webcam is connected.
2. Download model_finetuned2.h5 and put in the same folder as the script.
(Link to trained model: https://drive.google.com/drive/folders/1ICBfesmnxnN2seWF7ttB-InrNFMVid6D).
3. Open terminal or command prompt.
4. Run: python prototype.py
5. Click "Start" in the window to begin, "Stop" to stop the webcam.

Requirements:
--------------
- Python 3.x
- tensorflow
- opencv-python
- numpy
- Pillow

Install them using:
pip install tensorflow opencv-python numpy Pillow

Note:
------
This is a prototype for academic use. Predictions may not always
be accurate depending on camera quality and lighting.

Author: Christa Tracy Mojikon
Date  : 28/07/2025
