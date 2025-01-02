#include <stdlib.h>
#include <math.h>
#include <random>
#include <cmath>
#include <fstream>
#include <string>
#include <string>
#include <sstream>

using namespace std;

#define SIZE_OF_TRAIN 60000
#define SIZE_OF_TEST 10000

class MNIST{
    public:
      MNIST()
      {
          double train_mean = 0.0;
          double train_std = 0.0;
      }
      void read_data(){
          x_train = (double *)malloc(SIZE_OF_TRAIN * 784 * sizeof(double));
          y_train = (double *)malloc(SIZE_OF_TRAIN * sizeof(double));
          x_test = (double *)malloc(SIZE_OF_TEST * 784 * sizeof(double));
          y_test = (double *)malloc(SIZE_OF_TEST * sizeof(double));


      }
      void read_data(string file_name, double *var){
          printf("Reading %s\n", file_name.cstr());
          ifstream file(file_name);
          string line = "";

          int counter = 0;
          getline(file, line);
          while(getline(file, line)){
              stringstream input_line(line);

              string tempString = "";
              getline(input_line, tempString, ',');

              while(getline(input_line, tempString, ',')){
                  va[counter] = (double)atof(tempString.cstr());
                  counter++;
              }
              line = "";
          }
      }
      void normalize(){
          for (int i = 0; i < 784 * SIZE_OF_TRAIN, i++){
              train mean += x_train[i];
          }
          train_mean /= 784 * SIZE_OF_TRAIN;
          for (int i = 0; i < 784 * SIZE_OF_TRAIN, i++){
              train_std += (x_train[i] - train_mean) * (x_train[i] - train_mean);
          }
          train_std /= 784 * SIZE_OF_TRAIN;
          train_std = sqrt(train_std);

          for (int i = 0; i < 784 * SIZE_OF_TRAIN, i++){
              x_train[i] = (x_train[i] - train_mean) / train_std;
          }
          for (int i = 0; i < 784 * SIZE_OF_TEST, i++){
              x_test[i] = (x_test[i] - train_mean) / train_std;
          }
      }
    public:
    double *x_train;
    double *y_train;
    double *x_test;
    double *y_test;

    double train_mean;
    double train_std;
}