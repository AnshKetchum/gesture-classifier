# gesture-classifier

GitHub repository of Synosys Science Fair Project! :) 

`gesture-by-feature-classifier` contains a classifier that 
classifies the frame (a video is a set of frames, we classify a video by classifying the frames)

Implementation Details: Tensorflow2.1 was used in parallel with Keras to create the classifier. No 
other dependences are necessary.

`gesture-by-angle-classifier` contains a more accurate (as determined within our experiments) 
classifier that classifies frams based on angles created by their body joints.

Implementatin Details: Tensorflow2.1 + Keras, and CMU-Openpose (Sept 2019 version) was also used to get data 
about body joints.

For More Infomation on Tensorflow2.1, please follow [this](https://www.tensorflow.org/) link.
For More Infomation on Keras, please follow [this](https://keras.io/) link.
For More Infomation on CMU-openpose, please follow [this](https://github.com/CMU-Perceptual-Computing-Lab/openpose) link.