#include "knn.h"

/****************************************************************************/
/* For all the remaining functions you may assume all the images are of the */
/*     same size, you do not need to perform checks to ensure this.         */
/****************************************************************************/


// Checks if the correct number of bytes are read from the file
void correct_read(int error, int read){
  if (error != read){
    fprintf(stderr, "Error while reading the file\n");
    exit(1);
  }
  return;
}


/** 
 * Return the euclidean distance between the image pixels (as vectors).
 */
double distance(Image *a, Image *b) {
  // TODO: Return correct distance
  double total_sum = 0;
  int max = a->sx * a->sy;
  for (int q = 0; q < max; q++){
    total_sum += pow((a->data[q] - b->data[q]), 2);
  }
  return sqrt(total_sum);
}

/**
 * Given the input training dataset, an image to classify and K,
 *   (1) Find the K most similar images to `input` in the dataset
 *   (2) Return the most frequent label of these K images
 * 
 * Note: If there's multiple images with the same smallest values, pick the
 *      ones that come first. For automarking we will make sure the K smallest
 *      ones are unique, so it doesn't really matter.
 */ 
int knn_predict(Dataset *data, Image *input, int K) {
  // TODO: Replace this with predicted label (0-9)
  // Data allocate for keep track of label, k_dist, and some tmp variables to keep track of sorting
  unsigned char label[K];    // Store K small label from the array
  double k_dist[K];   // Store the index or position
  double tmp_dist, tmp_label, current_dist, current_label;
  int tmp_count = 0, count = -1, ret_label = 0;

  // Calculate the first K index and store them in k_dist array
  for (int i = 0; i < K; i++){
    k_dist[i] = distance((data->images + i), input);
    label[i] = data->labels[i];
  }

  // Calculate the distance of single image against the whole data starting from the K
  for (int i = K; i < data->num_items; i++){
    current_dist = distance((data->images + i), input);
    current_label = data->labels[i];

    // Finding the small K set of dist
    for (int j = 0; j < K; j++){
      if (current_dist < k_dist[j]){
        // Store the first value that is less then the existing dist in k_dist array
        tmp_dist = k_dist[j];
        tmp_label = label[j];
        
        // Assign the new smaller value inside the K dist array
        k_dist[j] = current_dist;
        label[j] = current_label;

        // Assign the removed value to check if it is less the the existing one
        current_dist = tmp_dist;
        current_label = tmp_label;
      }
    }
  }

  // Found the popular label in the K array set
  for (int i = 0; i < K; i++){
    tmp_count = 0;
    tmp_label = label[i];

    for (int j = 0; j < K; j++){
      if (tmp_label == label[j]){
        tmp_count++;
      }
    }

    // Store the popular label and verify it
    if (tmp_count > count){
      ret_label = tmp_label;
      count = tmp_count;
    }
  }
  
  return ret_label;
}


/**
 * This function takes in the name of the text file containing the image names.
 * Reads the data from the file assume that the image size is the same and stores it
 * by allocating dynamic memory. This also stores the correct label of the image.
 */
Dataset *load_dataset(const char *filename) {
  // TODO: Allocate data, read image data / labels, return
  FILE* file_type = fopen(filename, "rb");
  Dataset *node = NULL;
  int error = 0;
  int xy = 28;

  // Check the file exist
  if (file_type == NULL){
    perror("fopen");
    exit(1);
  }
  
  // Allocate Memory Dataset
  node = calloc(sizeof(Dataset), 1);
  // Check if there is enough memory
  if (node == NULL){
    fprintf(stderr, "Not enough memory available\n");
    exit(1);
  }

  // Read the total number of images in the file
  error = fread(&node->num_items, sizeof(int), 1, file_type);
  correct_read(error, 1);
  
  // Allocate Memory label, images
  node->images = calloc(sizeof(Image), node->num_items);
  node->labels = calloc(sizeof(unsigned char), node->num_items);
  // Check if there is enough memeory
  if (node->images == NULL || node->labels == NULL){
    fprintf(stderr, "Not enough memory available\n");
    exit(1);
  }

  // Traverse through the file and store the pixels in the image data array
  for (int i = 0; i < node->num_items; i++){
    // Read the ith label of the image in the file
    error = fread((node->labels + i), sizeof(unsigned char), 1, file_type);
    correct_read(error, 1);
    
    // Read the data of the ith image and store it
    node->images[i].sx = xy;
    node->images[i].sy = xy;
    node->images[i].data = calloc(sizeof(unsigned char), xy*xy);

    if (node->images[i].data == NULL){
      fprintf(stderr, "Not enough memory available\n");
      exit(1);
    }
    error = fread(node->images[i].data, sizeof(unsigned char), xy*xy, file_type);
    correct_read(error, xy*xy);
  }

  // Close the file and return the node
  fclose(file_type);
  return node;
}

/**
 * Free all the allocated memory for the dataset
 */
void free_dataset(Dataset *data) {
  // TODO: free data
  for (int i = 0; i < data->num_items; i++){
    free(data->images[i].data);
  }

  free(data->images);
  free(data->labels);
  free(data);
}



/**
 * This function should be called by each child process, and is where the 
 * kNN predictions happen. Along with the training and testing datasets, the
 * function also takes in 
 *    (1) File descriptor for a pipe with input coming from the parent: p_in
 *    (2) File descriptor for a pipe with output going to the parent:  p_out
 * 
 * Once this function is called, the child should do the following:
 *    - Read an integeages `start_idx` to `start_idx+N-1`
 *    - Write an integer representir `start_idx` from the parent (through p_in)
 *    - Read an integer `N` from the parent (through p_in)
 *    - Call `knn_predict()` on testing imng the number of correct predictions to
 *        the parent (through p_out)
 */
void child_handler(Dataset *training, Dataset *testing, int K, 
                   int p_in, int p_out) {
  // TODO: Compute number of correct predictions from the range of data 
  //      provided by the parent, and write it to the parent through `p_out`.
  
  // Initialize the fields used in computation AND read the start_idx and number of images
  int start_idx = 0, N_images = 0, check = 0, total_correct = 0, error = 0;
  error = read(p_in, &start_idx, sizeof(int));
  check = read(p_in, &N_images, sizeof(int));
  
  // Check if the read fails
  if (error < 0 || check < 0){
    perror("Read to pipe");
  }
  check = 0;
  
  // Calculate the number of times the correct label was found
  for (int i = start_idx; i < (start_idx + N_images); i++){
    check = knn_predict(training, (testing->images + i), K);
    if (check == testing->labels[i]){
      total_correct++;
    }
  }

  // Check labels and write them in the pipe
  if(write(p_out, &total_correct, sizeof(int)) == -1){
    perror("Write to pipe");
  }
  return;
}