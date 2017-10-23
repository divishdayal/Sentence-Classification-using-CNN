# Sentence-Classification-using-CNN

This implements the Code for the paper Convolutional Neural Networks for Sentence Classification (EMNLP 2014).
Runs the model on Pang and Lee's movie review dataset (MR in the paper). The code is implemented in tensorflow and datasets have been
included in the root directory as pos.txt and neg.txt

accuracy of the model - ~78%

## Run
To run the model - 
> python model.py

## Model details 
 * 3 convolution/filter sizes(3,4,5) ;
 * 30 filters height ;
 * dropout-0.4 ;
 * word vecs - pretrained, Glove, 50-dimensions;
 * Adam optimizer ; 
 * learning rate - 0.0005

Also, Batch Normalisation code has been written, but commented out as in this case, it didn't improve on the results without it.

Most of the hyper-parameters can be changed in the config.yml file.

