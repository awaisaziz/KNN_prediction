#include "knn.h"

// Makefile included in starter:
//    To compile:               make
//    To decompress dataset:    make datasets
//
// Example of running validation (K = 3, 8 processes):
//    ./classifier 3 datasets/training_data.bin datasets/testing_data.bin 8

/*****************************************************************************/
/* This file should only contain code for the parent process. Any code for   */
/*      the child process should go in `knn.c`.						         */
/*****************************************************************************/

/**
 * main() takes in 4 command line arguments:
 *   - K:  The K value for kNN
 *   - training_data: A binary file containing training image / label data
 *   - testing_data: A binary file containing testing image / label data
 *   - num_procs: The number of processes to be used in validation
 * 
 * You need to do the following:
 *   - Parse the command line arguments, call `load_dataset()` appropriately.
 *   - Create the pipes to communicate to and from children
 *   - Fork and create children, close ends of pipes as needed
 *   - All child processes should call `child_handler()`, and exit after.
 *   - Parent distributes the testing set among childred by writing:
 *        (1) start_idx: The index of the image the child should start at
 *        (2)    N:      Number of images to process (starting at start_idx)
 *     Each child should gets N = ceil(test_set_size / num_procs) images
 *      (The last child might get fewer if the numbers don't divide perfectly)
 *   - Parent waits for children to exit, reads results through pipes and keeps
 *      the total sum.
 *   - Print out (only) one integer to stdout representing the number of test 
 *      images that were correctly classified by all children.
 *   - Free all the data allocated and exit.
 */
int main(int argc, char *argv[]) {
  // TODO: Handle command line arguments
  if (argc != 5){
    fprintf(stderr, "Usage: %s K training_lists testing_lists processes\n", argv[0]);
    return 1;
  }

  // Dataset of images initial set up and tree
  Dataset *training_list = NULL;
  Dataset *testing_list = NULL;
  int K = atoi(argv[1]);
  // create children, pipes
  int num_procs = atoi(argv[argc-1]);
  int child_in_pipes[num_procs][2];  // parent to child info
  int parent_in_pipes[num_procs][2]; // Child to parent info
  int child_procs[num_procs];

  // Load the dataset from the files
  training_list = load_dataset(argv[2]);
  testing_list = load_dataset(argv[3]);

  // Check if there is enough memory available to store the testing list
  if (training_list == NULL && training_list == NULL && K <= 0 && num_procs <= 0){
    fprintf(stderr, "Not enough memory to store the testing list or K/num_process value is neg/zero\n");
    return 1;
  }

  // TODO: Spawn `num_procs` children and pipes
  // Creating all the required pipes in Parent
  for (int i = 0; i < num_procs; i++){
    if (pipe(child_in_pipes[i]) == -1 || pipe(parent_in_pipes[i]) == -1){
      perror("pipe error");
      exit(1);
    }
  }

/****************************************** CHILDREN ****************************/
  // Create num_procs children
  for (int i = 0; i < num_procs; i++){
    child_procs[i] = fork();
    if (child_procs[i] == -1){
      perror("fork failed");
      exit(1);
    }
    // Successfully forked the child
    if (child_procs[i] == 0){ // Child process begins
      // Close all the pipes inside the child process of other child
      for (int j = 0; j < num_procs; j++){
        if (i == j){
          close(child_in_pipes[i][1]); // close child pipe write
          close(parent_in_pipes[i][0]); // close parent pipe read
        } else {
          close(child_in_pipes[j][0]); // Close all the other child's and parent's read and write
          close(child_in_pipes[j][1]);
          close(parent_in_pipes[j][0]);
          close(parent_in_pipes[j][1]);
        }
      }
      child_handler(training_list, testing_list, K, child_in_pipes[i][0], parent_in_pipes[i][1]);
      close(child_in_pipes[i][0]); // Close the read end of the child_in
      close(parent_in_pipes[i][1]); // Close the write end of the pipe of parent_in
      // Free allocated memory in Child
      free_dataset(training_list);
      free_dataset(testing_list);
      exit(0);  // Child process ends
    }
  }

/************************************** Child/Parent read/Write pipes ****************************/
  // Close all the remaining read pipes in Child_in and write end in Parent_in
  for (int i = 0; i < num_procs; i++){
    close(child_in_pipes[i][0]); // Close all the read end of the pipe in Child_in
    close(parent_in_pipes[i][1]); // Close all the write end of pipe in Parent_in
  }

/************************************* Information from Parent to Children **************************/
  // TODO: Send information to children
  // Starting index, total_size and the error while writing to the pipe
  int start_idx = 0, total_size = (int) ceil((double) testing_list->num_items / (double) num_procs);
  int tmp = 0, error1 = 0, error2 = 0;
  for (int i = 0; i < num_procs; i++){
    // A check for the last child
    if (i == num_procs - 1 && start_idx < testing_list->num_items){ // Last Child if fewer images or more images
      error1 = write(child_in_pipes[i][1], &start_idx, sizeof(int));
      total_size = testing_list->num_items - start_idx; // update the n_images for the final child
      error2 = write(child_in_pipes[i][1], &total_size, sizeof(int));
    } else if (start_idx >= testing_list->num_items) {
      // Check if the start index is greater than the total number of images then pass 0
      error1 = write(child_in_pipes[i][1], &tmp, sizeof(int));  // Write starting index
      error2 = write(child_in_pipes[i][1], &tmp, sizeof(int)); // Write N_images to process
    } else {
      error1 = write(child_in_pipes[i][1], &start_idx, sizeof(int));  // Write starting index
      error2 = write(child_in_pipes[i][1], &total_size, sizeof(int)); // Write N_images to process
    }

    start_idx += total_size;
    if (error1 == -1 || error2 == -1){
      perror("Write to pipe");
    }
  }

/********************************* Get info from the children ***********************/
  // TODO: Compute the total number of correct predictions from returned values
  // Close writing end Child_in and Reading end Parent_in while get info from the child
  int total_correct = 0, child_read = 0;
  for(int i = 0; i < num_procs; i++){
    close(child_in_pipes[i][1]); // Close all the write end of the pipe in Child_in
    // Read from the Parent_in file descriptor
    if(read(parent_in_pipes[i][0], &child_read, sizeof(int)) < 0){
      perror("Read to pipe");
    }
    total_correct += child_read;
    close(parent_in_pipes[i][0]); // Close all the read end of the pipe in Parent_in
  }

  // Print out final answer
  printf("%d\n", total_correct);

  // Free allocated memory in Parent
  free_dataset(training_list);
  free_dataset(testing_list);
  return 0;
}
