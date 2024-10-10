#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#define MAX_PREVS 3
#define MAX_ARGS 5
#define MAX_PARAM_TENSORS 10
// op codes
#define MATMUL 0
#define MEAN 1
#define MUL 2
#define RELU 3
#define LOGSOFTMAX 4

typedef struct {
    float* values;
    int* shape;
    int* strides;
    int ndim;
    int size;
} Arr;

typedef union {
    int ival;
    float fval;
    int* ilist;
} Arg;

typedef struct Tensor {
    Arr* data;
    Arr* grad;
    int op; // op used to create this tensor
    struct Tensor* prevs[MAX_PREVS]; // tensors that were processed by the op
    int num_prevs;
    Arg args[MAX_ARGS]; // additional args for the op (e.g. axis, stride etc.)
} Tensor;

Arr* create_arr(float* data, int* shape, int ndim);
Arr* create_arr_zeros(int *shape, int ndim);
void free_arr(Arr* a);
Tensor* create_zero_tensor(int* shape, int ndim);
Tensor* create_tensor(float* data, int* shape, int ndim);
void free_tensor(Tensor* t);
void backward(Tensor* t);

Tensor* mul(Tensor* a, Tensor* b);
void mul_backward(Tensor* out);
Tensor* mean(Tensor* a);
void mean_backward(Tensor* out);
Tensor* matmul(Tensor* a, Tensor* b);
void matmul_backward(Tensor* out);
Tensor* logsoftmax(Tensor* inp);
void logsoftmax_backward(Tensor* out);
Tensor* relu(Tensor* inp);
void relu_backward(Tensor* out);
void print_tensor(Tensor* t);

float random_normal() {
    float u1 = (float)rand() / RAND_MAX;
    float u2 = (float)rand() / RAND_MAX;
    return sqrtf(-2 * logf(u1)) * cosf(2 * M_PI * u2);
}

float rand_float() {
    return (float)rand() / (float)RAND_MAX;
}

float rand_range(float min, float max) {
    return min + rand_float() * (max - min);
}
// kaiming uniform initialization
float kaiming_uniform(int fan_in) {
    float gain = sqrtf(2.0f);  // for ReLU activation
    float std = gain / sqrtf(fan_in);
    float bound = sqrtf(3.0f) * std;
    return rand_range(-bound, bound);
}

// kaiming initialization
float kaiming_init(int fan_in) {
    float std_dev = sqrtf(2.0f / fan_in);
    return random_normal() * std_dev;
}

void print_tensor(Tensor* t) {
    printf("Tensor(\n");
    printf("\tdata: ");
    for (int i = 0; i < t->data->size; i++) printf("%f,", t->data->values[i]);
    printf("\n\tshape: ");
    for (int i = 0; i < t->data->ndim; i++) printf("%d,", t->data->shape[i]);
    printf("\n\tgrad: ");
    for (int i = 0; i < t->data->size; i++) printf("%f,", t->grad->values[i]);
    printf("\n)\n");
}

Arr* create_arr(float* data, int* shape, int ndim) {
    Arr* arr = create_arr_zeros(shape, ndim);
    memcpy(arr->values, data, arr->size * sizeof(float));
    return arr;
}

Arr* create_arr_zeros(int* shape, int ndim) {
    Arr* arr = (Arr*) malloc(sizeof(Arr));
    if (!arr) return NULL;

    arr->ndim = ndim;
    arr->shape = (int*) malloc(ndim * sizeof(int));
    if (!arr->shape) {
        free(arr);
        return NULL;
    }
    memcpy(arr->shape, shape, ndim * sizeof(int));

    arr->strides = (int*) malloc(ndim * sizeof(int));
    if (!arr->strides) {
        free(arr->shape);
        free(arr);
        return NULL;
    }

    arr->size = 1;
    for (int i = ndim - 1; i >= 0; i--) {
        arr->strides[i] = arr->size;
        arr->size *= shape[i];
    }

    arr->values = (float*) calloc(arr->size, sizeof(float));
    if (!arr->values) {
        free(arr->strides);
        free(arr->shape);
        free(arr);
        return NULL;
    }

    return arr;
}

Tensor* create_tensor(float* data, int* shape, int ndim) {
    Arr* d = create_arr(data, shape, ndim);
    Tensor* t = (Tensor*)malloc(sizeof(Tensor));
    t->data = d;
    t->grad = create_arr_zeros(shape, ndim);
    t->op = -1;
    t->num_prevs = 0;
    return t;
}

Tensor* create_zero_tensor(int* shape, int ndim) {
    Arr* d = create_arr_zeros(shape, ndim);
    Tensor* t = (Tensor*)malloc(sizeof(Tensor));
    t->data = d;
    t->grad = create_arr_zeros(shape, ndim);
    t->op = -1;
    t->num_prevs = 0;
    return t;
}

void free_arr(Arr* a) {
    if (a == NULL) return;
    if (a->values != NULL) {
        free(a->values);
    }
    if (a->shape != NULL) {
        free(a->shape);
    }
    if (a->strides != NULL) {
        free(a->strides);
    }
    free(a);
}

void free_tensor(Tensor* t) {
    if (t == NULL) return;
    if (t->data != NULL) free_arr(t->data);
    if (t->grad != NULL) free_arr(t->grad);
    free(t);
}

void backward(Tensor* t) {
    // assumes that the grad of `t` has been computed
    // and computes the grad for tensors in `t->prevs`
    // then calls the backward function on prev tensors
    if (t->op == MUL) {
        mul_backward(t);
    } else if (t->op == MEAN) {
        mean_backward(t);
    } else if (t->op == MATMUL) {
        matmul_backward(t);
    } else if (t->op == RELU) {
        relu_backward(t);
    } else if (t->op == LOGSOFTMAX) {
        logsoftmax_backward(t);
    }
    for (int i = 0; i < t->num_prevs; i++) {
        backward(t->prevs[i]);
    }
}

Tensor* mul(Tensor* a, Tensor* b) {
    Tensor* t = create_zero_tensor(a->data->shape, a->data->ndim);
    for (int i = 0; i < a->data->size; i++) {
        t->data->values[i] = a->data->values[i] * b->data->values[i];
    }
    t->op = MUL;
    t->num_prevs = 2;
    t->prevs[0] = a;
    t->prevs[1] = b;
    return t;
}

void mul_backward(Tensor* out) {
    for (int i = 0; i < out->data->size; i++) {
        out->prevs[0]->grad->values[i] += out->grad->values[i] * out->prevs[1]->data->values[i];
        out->prevs[1]->grad->values[i] += out->grad->values[i] * out->prevs[0]->data->values[i];
    }
}

Tensor* mean(Tensor* t) {
    Tensor* m = create_zero_tensor((int[]){1}, 1);
    float s = 0.0f;
    for(int i = 0; i < t->data->size; i++) s += t->data->values[i];
    m->data->values[0] = s/t->data->size;
    m->op = MEAN;
    m->num_prevs = 1;
    m->prevs[0] = t;
    return m;
}

void mean_backward(Tensor* out) {
    for (int i = 0; i < out->prevs[0]->grad->size; i++) {
        out->prevs[0]->grad->values[i] += out->grad->values[0] / out->prevs[0]->data->size;
    }
}

Tensor* logsoftmax(Tensor* inp) {
    // inp and out are both of shape (B,C)
    Tensor* t = create_zero_tensor(inp->data->shape, inp->data->ndim);
    for (int b = 0; b < inp->data->shape[0]; b++) {
        float maxv = inp->data->values[b * inp->data->strides[0]];
        for (int c = 1; c < inp->data->shape[1]; c++) {
            int pos = b * inp->data->strides[0] + c * inp->data->strides[1];
            if (maxv < inp->data->values[pos]) {
                maxv = inp->data->values[pos];
            }
        }
        float sumexp = 0.0f;
        for (int c = 0; c < inp->data->shape[1]; c++) {
            int pos = b * inp->data->strides[0] + c * inp->data->strides[1];
            float expval = expf(inp->data->values[pos] - maxv);
            sumexp += expval;
        }
        for (int c = 0; c < inp->data->shape[1]; c++) {
            int pos = b * inp->data->strides[0] + c * inp->data->strides[1];
            t->data->values[pos] = inp->data->values[pos] - maxv - logf(sumexp);
        }
    }
    t->op = LOGSOFTMAX;
    t->num_prevs = 1;
    t->prevs[0] = inp;
    return t;
}
void logsoftmax_backward(Tensor* out) {
    // out is of shape (B,C)
    for (int b = 0; b < out->data->shape[0]; b++) {
        float gradsum = 0.0f;
        for (int c = 0; c < out->data->shape[1]; c++) {
            gradsum += out->grad->values[b * out->grad->shape[1] + c];
        }
        for (int c = 0; c < out->data->shape[1]; c++) {
            int pos = b*out->data->shape[1] + c;
            out->prevs[0]->grad->values[pos] += out->grad->values[pos] - expf(out->data->values[pos]) * gradsum;
        }
    }
}

Tensor* relu(Tensor* inp) {
    Tensor* t = create_zero_tensor(inp->data->shape, inp->data->ndim);
    for (int i = 0; i < inp->data->size; i++) {
        t->data->values[i] = (inp->data->values[i] > 0) ? inp->data->values[i] : 0;
    }
    t->op = RELU;
    t->num_prevs = 1;
    t->prevs[0] = inp;
    return t;
}

void relu_backward(Tensor* out) {
    for (int i = 0; i < out->data->size; i++) {
        out->prevs[0]->grad->values[i] = (out->prevs[0]->data->values[i] > 0) ? out->grad->values[i] : 0;
    }
}

Tensor* matmul(Tensor* a, Tensor* b) {
    // (P,Q) x (Q,R) = (P,R)
    int P = a->data->shape[0];
    int Q = a->data->shape[1];
    int R = b->data->shape[1];
    Tensor* t = create_zero_tensor((int[]) {P, R}, 2);
    #pragma omp parallel for
    for (int i = 0; i < P; i++) {
        for (int j = 0; j < R; j++) {
            float tmp = 0.0f;
            for (int k = 0; k < Q; k++) {
                int pos_a = i * a->data->strides[0] + k * a->data->strides[1];
                int pos_b = k * b->data->strides[0] + j * b->data->strides[1];
                tmp += a->data->values[pos_a] * b->data->values[pos_b];
            }
            int pos_c = i * R + j;
            t->data->values[pos_c] = tmp;
        }
    }
    t->op = MATMUL;
    t->num_prevs = 2;
    t->prevs[0] = a;
    t->prevs[1] = b;
    return t;
}


void matmul_backward(Tensor* out) {
    // a (P,Q), b (Q,R), c (P, R)
    int P = out->prevs[0]->data->shape[0];
    int Q = out->prevs[0]->data->shape[1];
    int R = out->prevs[1]->data->shape[1];
    
    // dc x b.T  (P,R) x (R,Q) => (P,Q)
    #pragma omp parallel for
    for (int i = 0; i < P; i++) {
        for (int j = 0; j < Q; j++) {
            float tmp = 0.0f;
            for (int k = 0; k < R; k++) {
                // (k,j) in b.T is (j,k) in b
                int pos_b = j * out->prevs[1]->data->strides[0] + k * out->prevs[1]->data->strides[1]; 
                tmp += out->grad->values[i * R + k] * out->prevs[1]->data->values[pos_b];
            }
            int pos_da = i * Q + j;
            out->prevs[0]->grad->values[pos_da] = tmp;
        }
    }
    
    // a.T x dc  (Q,P) x (P,R) => (Q,R)
    #pragma omp parallel for
    for (int i = 0; i < Q; i++) {
        for (int j = 0; j < R; j++) {
            float tmp = 0.0f;
            for (int k = 0; k < P; k++) {
                // (i,k) in a.T is (k,i) in a
                int pos_a = k * out->prevs[0]->data->strides[0] + i * out->prevs[0]->data->strides[1]; 
                tmp += out->grad->values[k * R + j] * out->prevs[0]->data->values[pos_a];
            }
            int pos_db = i * R + j;
            out->prevs[1]->grad->values[pos_db] = tmp;
        }
    }   
}

