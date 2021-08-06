#pragma once

#include <math.h>    // Need this for sqrt()
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>       // fork()
#include <sys/types.h>    // pid_t
#include <sys/wait.h>     // wait()

/* This struct stores the data for an image */
typedef struct {
    int sx;               // x resolution
    int sy;               // y resolution
    unsigned char *data;  // List of `sx * sy` pixel color values [0-255]
} Image;

/* This struct stores the images / labels in the dataset */
typedef struct {
    int num_items;          // Number of images in the dataset
    Image *images;          // List of `num_items` Image structs
    unsigned char *labels;  // List of `num_items` labels [0-9]
} Dataset;

// These are all the same as A1.
double distance(Image *a, Image *b);
int knn_predict(Dataset *data, Image *img, int K);

// This is the same as A2.
Dataset *load_dataset(const char *filename);
void free_dataset(Dataset *data);

// New for A3!
void child_handler(Dataset *training, Dataset *testing, int K, int p_in, int p_out);
