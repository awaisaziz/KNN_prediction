# KNN_prediction

K-Nearest_Neighbour (KNN) is a simple machine learning technique for classification of labels (0 - 9). The basic idea is to label a new image by finding the k most similar images in the training set and assign the new image the label that is most common among those k similar images. The code takes two file sample images that are already correctly labelled (training set) and try to extract knowledge from it that it can use the to later label the new images (Called test set) that it has not yet seen. These files are in the binary format which contains 28 by 28 dimension (pixel) of label (image).

It contains a Makefile file which is helpful when run on linux. It also contains some the linux system programming to add parallelism to the program which increases the computational power and perform the task at a higher cost (performance) and less time. This can be run on linux environment.
