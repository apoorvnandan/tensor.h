#include "tensor.h"

Tensor* model(Tensor* inp, Tensor* w1, Tensor* w2) {
    printf("forward pass\n"); 
    Tensor* res = inp;
    for (int l = 0; l < 2; l++) {
        printf("layer %d\n", l);
        if (l == 0)
            res = matmul(res, w1);
        else
            res = matmul(res, w2);
        print_tensor(res);
    }
    return res;
}

int main() {
    Tensor* inp = create_zero_tensor((int[]){2,2},2);
    for (int i = 0; i < 4; i++)
        inp->data->values[i] = (float)i;
    Tensor* w1 = create_zero_tensor((int[]){2,2}, 2);
    Tensor* w2 = create_zero_tensor((int[]){2,2}, 2);

    for (int i = 0; i < w1->data->size; i++) w1->data->values[i] = kaiming_uniform(784);
    for (int i = 0; i < w2->data->size; i++) w2->data->values[i] = kaiming_uniform(128);
    printf("w1:\n");
    print_tensor(w1);
    printf("w2:\n");
    print_tensor(w2);
    
    Tensor* out = model(inp, w1, w2);
    printf("out\n");
    print_tensor(out);
    printf("w1o\n");
    print_tensor(out->prevs[0]);
    printf("w2\n");
    print_tensor(out->prevs[1]);
    printf("inp\n");
    print_tensor(out->prevs[0]->prevs[0]);
    printf("w1\n");
    print_tensor(out->prevs[0]->prevs[1]);

    return 0;
}

