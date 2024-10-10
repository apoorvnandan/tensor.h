# Creating a tiny tensor library in C

Documenting my pov while building this. The objective is to help people understand neural networks from absolute scratch. No pytorch. No numpy. Just maths and C.

Should be readable for anyone who knows programming **even if you are not familiar with machine learning or neural networks at all**. 

<img width="1383" alt="image" src="https://github.com/user-attachments/assets/f0de68cd-dc7b-4592-b68a-265793c2c6f9">

Moreover, building this from scratch, and in C, does not mean that the code and the APIs will not be user friendly. In fact, we'll create the needed abstractions and show just how easy it is to code and train different neural network architectures with this tiny library.

## Contents:

* What are neural networks?
* Creating tensors
* Defining a loss function
* Optimising the loss - Autograd
* Implementing operations
* Training neural networks

## What are neural networks?

Think about the process of coding a function. For some tasks, functions can be straightforward to code.

Here's an example: "Write a function to change a color image to grayscale."

There is a clear set of instructions (`for each pixel: change RGB value to new_value = 0.299R + 0.587G + 0.114B`) that you can code in your favorite programming language and create a function that will solve this task. The function will be determinisitic, giving you exact predictable outputs for your input images.

There are other tasks, where it's pretty much impossible to come up with a set of instructions needed to get the output from the input. And therefore, you cannot write the code for them.

Example: "The input image contains either a cat or a dog. Write a function to output `CAT` or `DOG` based on the image contents."

<img width="268" alt="image" src="https://github.com/user-attachments/assets/18a1c0a0-5cbf-407d-8598-0cf4c0f58f06">

Think about the code you can write for this. You'll quickly realise that there is no specific set of instructions you can code here, to create this function. You can, however, write a special kind of a function that can solve this task. Let's write one such special function. The code is in C, but the logic should be readable by anyone.

```c
int cat_or_dog(float* input_img, float* w1, float* w2) {
    float* x1 = matrix_multiplication(input_img, w1);
    float* x2 = relu(x1);
    float* x3 = matrix_multiplication(x2, w2);
    float* x4 = logsoftmax(x3);
    if (x4[0] < x4[1]) return 0; // indiciating a "cat"
    return 1; // indicating a "dog"
}
```

You'll notice that this function is a bit weird. First of all it takes two other float arrays `w1` and `w2` as inputs, apart from the image. (The image here is a grayscale picture where each pixel value (0-255) is divided by 255 to represent it as a large float array)

Then, the function proceeds to do some weird mathematical operations on the image.

* `matrix_multiplication` is self explanatory.
* `relu` is a mathematical operation, equivalent of max(x, 0) on every number within the array.
* `logsoftmax` is a another mathematical function, and I'll just write the formula below, because it's not important for the point I'm trying to make at the moment.

$$
\text{log-softmax}(x_i) = \log \left( \frac{e^{x_i}}{\sum_{j} e^{x_j}} \right)
$$

The weirdest thing though, is that this function, for very specific values of `w1` and `w2`, will actually give you the correct output for like 99% of images! If you're completely new to machine learning and neural networks, you might be surprised. But defining these functions and finding the optimal values for `w1`, `w2`, etc. is pretty much what deep learning is all about.

Functions like these are called neural networks. And the additional inputs like `w1` and `w2` are called parameters or weights. When you "train" a neural network, you find the "correct" values of these parameters for the task you're trying to solve. 

You've probably heard about, or used ChatGPT, and other similar AI assistants. They are powered by neural networks as well. They generate their response to your message, by converting your text into some numbers, and then doing these weird mathematical operations between them and the parameters of the network to output the next word (or part of the word). Then they do this until the entire response is generated! Anyhow, we'll do something similar from scratch and build a nice little library to code different neural networks for different tasks, and train them.

## Creating tensors

Okay, I'll assume you are familiar with 1D arrays, 2D arrays, etc. 
We will work with N-D arrays here. And we will call these N-D arrays tensors.

To work with N-D arrays, we will create a `struct` called `Arr` here.

```c
typedef struct {
    float* values;
    int* shape;
    int* strides;
    int ndim;
    int size;
} Arr;
```

This struct holds everything needed. All the values are inside the 1D values array. Shape should be obvious. Strides are something that's really useful for some operations, and I'll explain them later. 

The key thing to understand here, is that as long as you know all these properties about an N-D array, you can perform pretty much any operation on it.

Here's an example:
To do matrix multiplication between two 2D arrays with shapes (4,2) and (2,3), we can write the following code.

```c
void matmul(Arr* c, Arr* a, Arr* b) {
    // (P,Q) x (Q,R) = (P,R)
    int P = a->shape[0];
    int Q = a->shape[1];
    int R = b->shape[1];
    for (int i = 0; i < P; i++) {
        for (int j = 0; j < R; j++) {
            float tmp = 0.0f;
            for (int k = 0; k < Q; k++) {
                int pos_a = i * a->strides[0] + k * a->strides[1];
                int pos_b = k * b->strides[0] + j * b->strides[1];
                tmp += a->values[pos_a] * b->values[pos_b];
            }
            int pos_c = i * c->strides[0] + j * c->strides[1];
            c->values[pos_c] = tmp;
        }
    }
}
```

Don't worry about this code, right now. We'll get to N-D array operations in the later sections.

## Defining a loss function

Going back to the initial example, our input, along with the parameters `w1` and `w2` are all instances of tensors. The challenge here is to find those specific values for `w1` and `w2` which make the function actually work.

In order to do that, we collect some data of accurate inputs and outputs for our function (the neural network). e.g. images of cats and dogs along with labels. Then, we define a different function, one that operates on the outputs of our neural network and the labelled outputs, and returns a score that represents how good or bad the neural network is.

Here is an example:

```c
float loss_fn(Arr* logsoftmax_outputs, Arr* labels) {
    // `outputs` is of shape (2) : 2 catgories to choose from
    // `labels` is of shape (2) : [1,0] for cat and [0,1] for dog
    float s = 0.0f;
    for (int i = 0; i < logsoftmax_outputs->size; i++) {
        s += a->data[i] * labels->data[i] * (-1);    
    }
    return s/a->size;
}
```

To understand what this function does, think about the following:
* The `logsoftmax_outputs` will range from `-infinity` to `0`. (See the `cat_or_dog` function above)
* What happens when `w1` and `w2` are far from the correct/optimal values, and the neural network function gives you random `logsoftmax_outputs`, say `[-0.69, -0.69]`
* Assume the label is `[0,1]` (i.e. `dog`), the loss value will be `0.34`.
* What happens when `w1` and `w2` have optimal values, such that the neural network gives you `logsoftmax_outputs` as `[-2.3, -0.01]`.
* Assuming the same labels (`[0,1]`), the loss value will be `0.005`.
 
What this should tell you, is that the loss value is close to zero when `w1` and `w2` have good values, and the neural network function is accurate.

The task of finding good values for the parameters `w1` and `w2` then becomes the task of finding `w1` and `w2` that produce low values from the loss function!

## Optimising the loss - Autograd

Remember differentials from your high school calculus? Well, the core concept is this:

* We want an answer to the question: How much will the loss value change if we change the values of `w1` and `w2` by some small amount (think ~0.001).
* To find this, we calculate the gradient of the loss value, w.r.t a parameter, say `w1` : $\frac{\partial L}{\partial w_1}$
* To calculate gradients like this, we need a generic procedure. Something we can apply regardless of the operations present within the neural network.

This procedure comes from chain rule.

Lets write down the relationship between w1 and the loss value.

$$
\text{w1-out} = \mathbf{x} \cdot \mathbf{w}_1 
$$

$$
\text{relu-out} = \max(\text{w1-out}, 0)
$$

$$
\text{w2-out} = \text{relu-out} \cdot \mathbf{w}_2
$$

$$
\text{logsofmax-out} = \text{logsoftmax}(\text{w2-out})
$$

$$
\text{L} = \text{mean}(\text{logsoftmax-out} \odot \text{labels})
$$

To find $\frac{\partial L}{\partial w_1}$ we need to apply the chain rule at each operation.

$$
\frac{\partial L}{\partial w_1} = \frac{\partial L}{\partial \text{logsoftmax-out}} \frac{\partial \text{logsoftmax-out}}{\partial \text{w2-out}} \frac{\partial \text{w2-out}}{\partial \text{relu-out}} \frac{\partial \text{relu-out}}{\partial \text{w1-out}} \frac{\partial \text{w1-out}}{\partial w_1}
$$

Each of these can be easily calculated individually. Why? Well there's a nice little set of operations that all neural networks are composed of. And the gradients for these operations are well known and easy to implement. e.g. for the matrix multiplication operation 

$$
C = A \cdot B
$$

The gradients can be calculated as:

$$
\text{dA} = \text{dC} \cdot B^{T}
$$

$$
\text{dB} = A^{T} \cdot \text{dC} 
$$

So to calculate the gradients of a parameter like `w1` w.r.t the loss, all we need to know of the set of operations from `w1` to the loss, as coded within the neural network.

Clarity comes with implementation, so let's do this programatically.

We will create a new struct:

```c
typedef union {
    int ival;
    float fval;
} Arg;

typedef struct Tensor {
    Arr* data;
    Arr* grad;
    int op; // op used to create this tensor
    struct Tensor* prevs[MAX_PREVS]; // tensors that were processed by the op
    int num_prevs;
    Arg args[MAX_ARGS]; // additional args for the op (e.g. axis, stride etc.)
} Tensor;
```

This is a wrapper around our previous struct for N-D arrays. We have the array, and then we add everything we need to be able to compute the gradients.

* `grad` stores the gradient - another N-D array of the same shape as the actual `data` within the tensor.
* `op` stores the operation which was used to create this tensor, if there was one. e.g. if we declare two tensors `input` and `w1` and add them to get a new tensor `x`. `x` has the op `ADD`. This is optional, as some tensors are simply declared and not calculated from other tensors.
* `prevs` is an array of pointers to other tensors that were processed by `op` e.g. `input` and `w1` are `prevs` for the tensor `x` in the example above.
* `args` contains additional arguments used within the op. e.g. if you were to calculate the mean of a tensor along a specific axis, the axis would be an additional argument.

The main point here, is that once you have all these attributes bundled together, it's really simple to implement the gradient calculations.

Let me explain with some simple operations.

```c
Tensor* a, b; // intitialised to some values
Tensor* mul_out = mul(a, b); // element wise multiplication
Tensor* loss = mean(mul_out); // L
```

And this is what we are trying to compute.

$$
\frac{\partial \text{loss}}{\partial a} = \frac{\partial \text{loss}}{\partial \text{mul-out}} \frac{\partial \text{mul-out}}{\partial a}
$$

Lets do this step by step.

Step 1: The gradient of `loss` tensor (that will be stored inside it's `grad` attribute) is simply `1` because if we change it's value by a small amount, say 0.001, the loss value will change by the exact same amount!! The gradient of `mul_out` tensor will be this part.
 
$$
\frac{\partial \text{loss}}{\partial a} = \underbrace{\frac{\partial \text{loss}}{\partial \text{mul-out}}}_{\text{this part}} \frac{\partial \text{mul-out}}{\partial a}
$$

At this step, we are looking at the `loss` tensor. The `op` here, is `MEAN` operation. The `prevs` contains just one tensor: `mul_out`. And `args` is empty. We will calculate the `grad` of the `prev` tensor(s) using this information. 

> This procedure used to calculate the `grad` of the inputs of an operation using the `grad` of the output of the operation, is called the backward function of that operation.

Let's say the inpuut to the mean operation is `inp` and the output is `out`.

The gradient w.r.t each value within `inp` will be: $\frac{1}{N}$ where $N$ is the number of values within `inp`.

So the backward function for the mean operation will be:

```c
void mean_backward(Tensor* out) {
    for (int i = 0; i < out->prevs[0]->grad->size; i++) {
        out->prevs[0]->grad->values[i] += out->grad->values[0] / out->prevs[0]->data->size;
    }
}
```

Step 2: The gradient of tensor `a` is this whole part

$$
\frac{\partial \text{loss}}{\partial a} = \underbrace{\frac{\partial \text{loss}}{\partial \text{mul-out}} \frac{\partial \text{mul-out}}{\partial a}}_{\text{this whole part}}
$$

The first part of this is something we have already computed using the backward function of `mean` operation.
The second part can be computed using the backward function of `MUL` operation.

```c
void mul_backward(Tensor* out) {
    for (int i = 0; i < out->data->size; i++) {
        out->prevs[0]->grad->values[i] += out->grad->values[i] * out->prevs[1]->data->values[i];
        out->prevs[1]->grad->values[i] += out->grad->values[i] * out->prevs[0]->data->values[i];
    }
}
```

And then we can multiply the result with `mul_out.grad` to get `a.grad`.

We will code a new function to handle this:

```c
void backward(Tensor* t) {
    // assumes that the grad of `t` has been computed
    // and computes the grad for tensors in `t->prevs`
    // then calls the backward function on prev tensors
    if (t->op == MUL) {
        mul_backward(t);
    } else if (t->op == MEAN) {
        mean_backward(t);
    }
    // add more ops here
    for (int i = 0; i < t->num_prevs; i++) {
        backward(t->prevs[i]);
    }
}
```

So, let's say we start with 2 or 3 tensors, do a bunch of operations on them and come up with one final loss tensor.
We can simply call `backward` on the loss tensor, and compute the gradients for every tensor that was involved in computing the loss tensor, including the initial tensors.

This procedure of automatically calculating gradients for all the involved tensors is called autograd.

## Implementing operations

Right now, the backward function can work backwards through `mean` and `mul` operations to calculate `grad` values for all the tensors involved.

To extend this with more operations, we need 3 things:

1. A functon for the operation, or the forward part of the operation.
2. A backward function for the operation.
3. Adding the backward function to the code of `void backward(Tensor* t);`

Both the forward and backward functions basically involve using shape and strides to operate on N-D arrays.

But what are strides?

Let's say you have a tensor of shape: `(4,2,2,3)`. 4 dimensions - `[0,1,2,3]`. And the values are all stored in a long 1D array.

When you move one step forward on a dimension, you need to move a certain number of steps forward in the 1D array where the values are stored.

For dimension 3, e.g. going from (1,0,1,1) to (1,0,1,2), you would move exactly 1 step. 

For dimension 0, however, going from (0,1,1,0) to (1,1,1,0) would take you 3 * 2 * 2 = 12 steps forward on the 1D values array.

<img width="901" alt="image" src="https://github.com/user-attachments/assets/84231c25-1824-49b5-8cfa-8f24d206c781">

Strides dictate the number of steps you will move forward on the 1D flattened values array when you move 1 step forward on any of the dimensions.

Each dimension has its own stride. Something which can be calculated using the shape, which contains the size of each dimension.

```c
int s = 1;
for (int i = ndim - 1; i >= 0; i--) {
    arr->strides[i] = s;
    s *= shape[i];
}
```

To find the value of a tensor at a position `(i,j,k, ...)` you simply multiply each index with the stride of that dimension, and then look up the value at that position in your values array.

Strides are really helpful when it comes to operations where the tensor needs to be reshaped and rearranged. e.g. a sliding window operation can be performed by simply changing the shape and stride values. You don't need to touch the values array at all.

<img width="766" alt="image" src="https://github.com/user-attachments/assets/6147b3f9-63e0-4a4a-9b0c-03aab1f8c2e1">

The entire point I'm trying to convey in this section is that you need stride and shape values along with the 1D array of items to be able to perform operations on N-D tensors. 

Here is an example of coding a matrix multiplication operation for two 2-D tensors.

```c
Tensor* matmul(Tensor* a, Tensor* b) {
    // (P,Q) x (Q,R) = (P,R)
    int P = a->data->shape[0];
    int Q = a->data->shape[1];
    int R = b->data->shape[1];
    Tensor* t = create_zero_tensor((int[]) {P, R}, 2);
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
```

And here are two more examples of operations which just turn out to be useful when creating these neural network functions. (more on this later)

```c
Tensor* logsoftmax(Arr* inp) {
    // inp and out are both of shape (B,C)
    float* d = (float*) malloc(inp->data->size * sizeof(float));
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
            d[pos] = inp->data->values[pos] - maxv - logf(sumexp);
        }
    }
    Tensor* t = create_tensor(d, inp->data->shape, inp->data->ndim);
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
            gradsum += out->grad->values[b * out->shape[1] + c];
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
```

## Training neural networks

Our tensor library is good enough to train some simple neural nets now! Let's go through one example. Remember, training is just the process of finding the optimal values for the parameters of the function.

Objective: Train a neural network to identify the digit (one of 0,1,2...9) present in the input image. The input image will contain the picture of a handwritten digit. To do this, we have a bunch of input images, which are already manually labelled. This is our training data. The image is the input to our function, and the label is what we can use to calculate the loss value. The data in this case is coming from the [MNIST dataset](https://yann.lecun.com/exdb/mnist/), which is famous for being the "hello world" of machine learning datasets.

Now, from this data, we have 60,000 labelled images. Each image has 28 x 28 = 784 pixels. Which we will arrange in a nice 784 sized 1-D tensor. Each pixel is simply an integer between 0 and 255, as these are grayscale images. We will divide all values by 255 to make them into a float between 0 and 1. Why? Let me explain the process first and then we'll get into these details.

![image](https://github.com/user-attachments/assets/84c33ab5-c4bc-43b6-b007-7218cdcc8a45)

Each label is just a number between 0 and 9, and we will turn them into 10 sized 1-D tensors. 10? Umm, it's much easier to show then tell:

```
0 -> [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
1 -> [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
2 -> [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
3 -> [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
4 -> [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
5 -> [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
6 -> [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
7 -> [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
8 -> [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
9 -> [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
```

Turning numbers into arrays like this is called one-hot encoding.

So now, we have pixels from 60,000 images which stack together will make a tensor of size (60000,784) and 60,000 labels which make a tensor of size (60000,10).

The training process is done in batches. The batch size is something you choose, along with a bunch of other settings before you start training. How you chosse these values is something beyond the scope of this article. So I'll just give you some good default values. Batch size will be 128.

This means that we will pick 128 values at random from the images and their corresponding labels to create two tensors:

`batch_x` of shape (128, 784)

`batch_y` of shape (128, 10)

Training process: 
1. We will define a neural network similar to the `cat_or_dog` example, along with a loss function.
2. We will select a random batch from our training data to create `batch_x` and `batch_y`.
3. We will apply the neural network followed by the loss function on the labelled input images
4. We will calculate the gradients for the parameters of our neural network using the `backward` funciton.
5. We will change the values of the paramaters based on the gradients (so that the loss value goes down)
6. Log the loss value and go back to step 2.
7. Once the loss value is sufficiently close to zero, we will stop this process and end up with an accurate neural network.

This entire process is repeated several times (several = few thousand) till the loss value comes down. We call this the training loop.

Here is what this looks like in code.

```c
for (int i = 0; i < 1000; i++) {
    get_random_batch(batch_x, batch_y, x, y, B);

    // these 4 lines are equivalent of calling our neural network function
    // on the arguments: `batch_x`, `w1` and `w2`.
    Tensor* w1_out = matmul(batch_x, w1);
    Tensor* relu_out = relu(w1_out);
    Tensor* w2_out = matmul(relu_out, w2);
    Tensor* lout = logsoftmax(w2_out);

    // these two lines are the equivalent of calling our (simple) loss function
    Tensor* mul_out = mul(lout, batch_y);
    Tensor* loss = mean(mul_out);

    // calculate the gradients
    loss->grad->values[0] = 1.0f;
    backward(loss);

    if (i % 100 == 0) {
        printf("batch: %d loss: %f \n", i, loss->data->values[0]);
    }

    // update the parameter values
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
```

You may notice that we have a new variable `lr` in the portion of the code that updates the parameter values.

`lr` stands for learning rate. This is another one of the settings you choose before you start your training. We set this to a small value typically. In this example, it is set to 0.005.

Okay, but what are the intial values of `w1` and `w2` - the parameters of our neural network. Well, you you intialise them with random numbers. However, random numbers created with a specific process. 

At this point, let me explain something about training neural networks, which makes this whole process a lot more experimental than you would expect.

In order to train a neural network for a given task, we have to make a lot of choices.

- The code inside neural net function, which is also known as the architecture, which includes how many parameters we have, and how we use them all with the input to calculate the output.
- The way we prepare our input tensors (e.g. dividing the pixels by 255)
- The way we initialise the parameter values.
- The learning rate.
- The way we update the parameter values using the learning rate. e.g. we can change the learning rate for each batch.
- The batch size.
- The criteria for stopping the training loop.

There can be more, but you get the idea. The training process for any neural network, is very brittle. It feels pretty magical when it works, but it only works for some specific choices. And there is a good amount of maths that tells you what these good choices are, but that is not something I will be going into here. Just know that experimenting with these choices is what you do to come up with a neural network that solved your task.

For this example, we have defined a specifc function, which is our neural network. We picked specifc operations like `logsoftmax`. We have picked a specific way to create the random numbers for parameters. We have picked a specific learning rate and batch size. And we have picked a specific way to update the parameter values at each iteration of the training loop. 

There are other choices that work as well. But what we have is a good default to demonstrate how this all works.

You can go through the entire code in the files `tensor.h` and `test.c` now. That's all. I hope you enjoyed reading!

In the next part, we'll go over some popular neural network architectures (i.e. function designs) and the intuitions behind them, and using CUDA to do speed up these operations by utilising GPUs!

# thank you

if you have any questions or feedback, you can DM me on twitter/X, or create an issue on this repo.

my twitter: [https://twitter.com/_apoorvnandan](https://twitter.com/_apoorvnandan)

if you want to support my work:

buy me a coffee: [https://buymeacoffee.com/apoorvn](https://buymeacoffee.com/apoorvn)
