# Peruvian Sign Language Interpretation Dataset

The Peruvian Sign Language Interpretation dataset is available for download through this AEC link.


# 1.PREPROCESSING #

Every step counts with their own readme file for more details. This readme file is an overview and map of all the different scripts and functions that you can find and do with the data. You can generate the most-heay data with the generateDataset.sh script or find it in this link:

You can also find the conda environment srts2.yml to import all the libraries that use captioning related to SRT files. You can also use the environment.yml to work with all the libraries for videos.


## VIDEO TO ONLY SQUARE (CROP) ##

This code takes the downloaded youtube video from "Aprendo en Casa" and fixes coordinates to crop the section in which the interpreter is performing. This square can later be learned, but for the moment is a fixed space.

- **Input:** Raw video downloaded from Youtube 
(./PeruvianSignLanguaje/Data/Videos/RawVideo)

- **Output:** Video segmented with fixed coordinates showing only the signer/interpreter
(./PeruvianSignLanguaje/Data/Videos/OnlySquare)

- **Code:** ./PeruvianSignLanguaje/1.Preprocessing/Video/crop_video.py


## RAW TRANSCRIPT TO PER LINE (aligned to audio) ##

- **Input:** Transcript all together with no enter between sentences. The text was written by volunteers in simple text format.
(./PeruvianSignLanguaje/Data/Transcripts/all_together)

- **Output:** Transcript with every sentence in a different line
(./PeruvianSignLanguaje/Data/Transcripts/per_line)

- **Code:** ./PeruvianSignLanguaje/1.Preprocessing/PERLINE/convertToLines.py


## PER LINE TO SRT (aligned to audio) ##

- **Input:** Transcripts arranged such as every sentence (ended with period, exclamation or question mark) is in one line
(./PeruvianSignLanguaje/Data/Transcripts/per_line)
- SRT downloaded from subtitle.to/ and manually modified to introduce punctuation marks
(./PeruvianSignLanguaje/Data/SRT/SRT_raw)

- **Output:** SRT organized by sentence (with time aligned and correct transcript sentence)
(./PeruvianSignLanguaje/Data/SRT/SRT_voice_sentences)

- **Code:** ./PeruvianSignLanguaje/1.Preprocessing/SRTs/convert_subtitle_sentence.py



## SEGMENT GESTURES (aligned to interpreter) ##

- **Input:** SRT with annotations in ELAN by sign
- Rawvideo
(./PeruvianSignLanguaje/Data/SRT/SRT_gestures)

- **Output:** Segmented already cropped video of interpreter in frames corresponding to each sign
(./PeruvianSignLanguaje/Data/Videos/Segmented_gestures)

- **Code:** ./PeruvianSignLanguaje/2.Segmentation/cropInterpreterBySRT.py (prev: segmentPerSRT.py)


## SEGMENT SIGN SENTENCES (aligned to interpreter)##

- **Input:**  SRT with annotations in ELAN by sign sentence
(./PeruvianSignLanguaje/Data/SRT/SRT_gestures_sentence)

- **Output:** Segmented already cropped video of interpreter in frames corresponding to each sign sentence
(./PeruvianSignLanguaje/Data/Videos/Segmented_gesture_sentence)

- **Code:** ./PeruvianSignLanguaje/2.Segmentation/cropInterpreterBySRT.py (prev:segmentPerSRT.py)

## (Optional) Resize and crop video according to person size ##

- **Input:**  videos obtained after process cropInterpreterBySRT.py in case you have 
(./PeruvianSignLanguaje/Data/Videos/Segmented_gesture_sentence)

- **Output:** Do an additional crop to the video to have the interpreter box
(./PeruvianSignLanguaje/Data/Videos/Segmented_gesture_sentence)

- **Code:** 1.Preprocessing\Video\resizeVideoByFrame.py

## VIDEO TO KEYPOINTS (aligned to interpreter) ##

- **Input:** Segmented sign or sign sentence
(./PeruvianSignLanguaje/Data/Videos/Segmented_gestures)

- **Output:** Segmented sign or sign sentence images with landmarks
- Pickle files with coordinates of each of the 33 points obtained from mediapipe library (holistic) to annotate landmarks
- Json files with coordinates of the keypoint landmarks
(./PeruvianSignLanguaje/Data/Keypoints)

- **Code:** ./PeruvianSignLanguaje/3.Translation/FrameToKeypoint/ConvertVideoToKeypoint.py

Parameters:
--image creates pickle files of images. However it is recommended not to use it
--input_path


#### PIPELINE USED FOR LREC ####
For the AEC dataset:
1. cropInterpreterBySRT.py
2. convertVideoToDict.py
3. Classification\ChaLearn


