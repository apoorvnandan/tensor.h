#include "tensor.h"
#include <time.h>
#include <sys/time.h>

void get_time(struct timeval *t) {
    gettimeofday(t, NULL);
}

void load_csv(Tensor* x, Tensor* y, char* filename) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        perror("Unable to open file");
        exit(1);
    }

    char line[10000]; // Assuming no line will be longer than this
    char *token;
    
    for(int b = 0; b < 60000; b++) {
        if(fgets(line, sizeof(line), file) != NULL) {
            token = strtok(line, ",");
            for(int i = 0; i < 28*28 + 10; i++) {
                if (token == NULL) {
                    fprintf(stderr, "CSV format error: not enough columns\n");
                    fclose(file);
                    exit(1);
                }
                if(i < 28*28) {
                    x->data->values[b * 28 * 28 + i] = atof(token);
                } else {
                    y->data->values[b * 10 + (i - 28*28)] = atof(token) * (-1.0f);
                }
                token = strtok(NULL, ",");
            }
        } else {
            fprintf(stderr, "Not enough data for the specified batch size.\n");
            break;
        }
    }

    fclose(file);
}

void get_random_batch(Tensor* batch_x, Tensor* batch_y, Tensor* x, Tensor* y, int B) {
    static int seeded = 0;
    if (!seeded) {
        srand(0);
        seeded = 1;
    }
    if (B > x->data->shape[0] || B > y->data->shape[0]) {
        // Handle error: batch size too large
        return;
    }
    int *used_indices = (int *)calloc(x->data->shape[0], sizeof(int));
    
    for(int i = 0; i < B; i++) {
        int index;
        do {
            index = rand() % x->data->shape[0];  // Select random index
        } while(used_indices[index]);  // Ensure index hasn't been used yet
        used_indices[index] = 1;  // Mark index as used

        // Copy data for x - now directly from 1D array of 784 elements
        for(int j = 0; j < 784; j++) {
            int x_index = index * x->data->strides[0] + j;
            int batch_x_index = i * batch_x->data->strides[0] + j;
            batch_x->data->values[batch_x_index] = x->data->values[x_index];
        }

        // Copy data for y - assuming y remains with 10 classes for one-hot encoding
        for(int k = 0; k < 10; k++) {
            int y_index = index * y->data->strides[0] + k * y->data->strides[1];
            int batch_y_index = i * batch_y->data->strides[0] + k * batch_y->data->strides[1];
            batch_y->data->values[batch_y_index] = y->data->values[y_index];
        }
    }

    free(used_indices);
}

int main() {
    Tensor* x = create_zero_tensor((int[]){60000,784}, 2);
    Tensor* y = create_zero_tensor((int[]){60000,10}, 2);

    load_csv(x, y, "mnist_train.csv");

    printf("loaded csv\n");
    
    Tensor* w1 = create_zero_tensor((int[]){784,128}, 2);
    Tensor* w2 = create_zero_tensor((int[]){128,10}, 2);

    for (int i = 0; i < w1->data->size; i++) w1->data->values[i] = kaiming_uniform(784);
    for (int i = 0; i < w2->data->size; i++) w2->data->values[i] = kaiming_uniform(128);

    int B = 128;
    float lr = 0.005;
    Tensor* batch_x = create_zero_tensor((int[]){B, 784}, 2);
    Tensor* batch_y = create_zero_tensor((int[]){B, 10}, 2);


    struct timeval start, end;
    double elapsed_time;
    get_time(&start);
    printf("Start Time: %ld.%06ld seconds\n", start.tv_sec, start.tv_usec);


    for (int i = 0; i < 5000; i++) {
        get_random_batch(batch_x, batch_y, x, y, B);

        Tensor* w1_out = matmul(batch_x, w1);
        Tensor* relu_out = relu(w1_out);
        Tensor* w2_out = matmul(relu_out, w2);
        Tensor* lout = logsoftmax(w2_out);
        Tensor* mul_out = mul(lout, batch_y);
        Tensor* loss = mean(mul_out);
        loss->grad->values[0] = 1.0f;
        backward(loss);

        if (i % 100 == 0) {
            printf("batch: %d loss: %f \n", i, loss->data->values[0]);
        }

        for (int i = 0; i < w1->data->size; i++) {
            w1->data->values[i] -= w1->grad->values[i] * lr;
            w1->grad->values[i] = 0.0f;
        }
        for (int i = 0; i < w2->data->size; i++) {
            w2->data->values[i] -= w2->grad->values[i] * lr;
            w2->grad->values[i] = 0.0f;
        }

        free_tensor(w1_out);
        free_tensor(relu_out);
        free_tensor(w2_out);
        free_tensor(lout);
        free_tensor(mul_out);
        free_tensor(loss);
    }

    get_time(&end);
    printf("End Time:   %ld.%06ld seconds\n", end.tv_sec, end.tv_usec);

    // Calculate elapsed time
    elapsed_time = (end.tv_sec - start.tv_sec) + 
                   (end.tv_usec - start.tv_usec) / 1e6;
    printf("Elapsed Time: %.6f seconds\n", elapsed_time);

    return 0;
}
