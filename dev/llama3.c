#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

typedef struct {
    float* values;
    int* shape;
    int* strides;
    int ndim;
    int size;
} Arr;

Arr* create_arr(float* data, int* shape, int ndim);
Arr* create_arr_zeros(int *shape, int ndim);
void free_arr(Arr* a);

Arr* rmsnorm(Arr* x, Arr* weight) {
    Arr* o = create_arr_zeros(x->shape, x->ndim);
    float ss = 0.0f;
    for (int j = 0; j < x->size; j++) {
        ss += x[j] * x[j];
    }
    ss /= size;
    ss += 1e-5f;
    ss = 1.0f / sqrtf(ss);
    // normalize and scale
    for (int j = 0; j < size; j++) {
        o->values[j] = weight->values[j] * (ss * x->values[j]);
    }
    return o;
}


Arr* matmul(Arr* x, Arr* w) {
    // W (d,n) @ x (n,) -> xout (d,)
    int n = x->shape[0];
    int d = w->shape[0];
    Arr* o = create_arr_zeros((int[]){d}, 1);
    int i;
    #pragma omp parallel for private(i)
    for (i = 0; i < d; i++) {
        float val = 0.0f;
        for (int j = 0; j < n; j++) {
            val += w->values[i * n + j] * x->values[j];
        }
        o->values[i] = val;
    }
    return o;
}

typedef struct {
    int dim;  // 4096
    int ffn_hidden_dim;
    int n_layers;   // 32
    int n_heads;    // 32
    int n_kv_heads; // 8
    int vocab_size; // 128256
    int seq_len;    // 8192
    Arr* key_cache[];   // layer -> (seq_len, kv_dim)
    Arr* val_cache[];   // layer -> (seq_len, kv_dim)
    Arr* att[];         // layer -> (n_heads, seq_len) buffer for attention
    Arr* token_embedding_table;   // (vocab_size, dim)
    Arr* rms_weight[];       // layer -> (dim)
    Arr* rms_bffn_weight[];  // layer -> (dim)
    Arr* wq[];          // layer -> (dim, n_heads * head_size) = (4096,4096)
    Arr* wk[];          // layer -> (dim, n_kv_heads * head_size) = (4096, 1024)
    Arr* wv[];          // layer -> (dim, n_kv_heads * head_size) = (4096, 1024)
    Arr* wo[];          // layer -> (n_heads * head_size, dim)
} LlamaModel;

Arr* llama_forward(int token, int pos, LlamaModel model) {
    int dim = model.dim;
    int head_size = dim / model.n_heads;
    int kv_dim = model.dim * model.n_kv_heads / model.n_heads;  // 1024
    float* row = model.token_embedding_table->values + token * dim;
    Arr* e = create_arr_zeros((int[]){dim},1);
    memcpy(e->values, row, dim*sizeof(float));

    Arr* k;
    Arr* v;
    float* att = (float*) malloc((pos+1)*sizeof(float));
    for (int l = 0; l < model.n_layers; l++) {
        Arr* x = rmsnorm(e, model.rms_weight[l]);  // (4096)
        int loff = l * model.seq_len + kv_dim;
        k = model.key_cache[l]; // (seq_len, kv_dim) <-> (seq_len, n_kv_heads, head_size)
        v = model.val_cache[l]; // (seq_len, kv_dim)

        Arr* q = matmul(x, wq); // (4096)
        Arr* k_token = matmul(x, wk);      // (1024)
        Arr* v_token = matmul(x, wv);      // (1024)

        // cache token K and V
        memcpy(k->values + pos * kv_dim, k_token->values, kv_dim * sizeof(float));
        memcpy(v->values + pos * kv_dim, v_token->values, kv_dim * sizeof(float));

        // RoPE
        for(int i = 0; i < model.n_heads; i++) {
            for (int j = 0; j < head_size; j++) {
                float freq = 1.0f / powf(500000.0f, (float) j / (float) head_size);
                float val = pos * freq;
                float fcr = cosf(val);
                float fci = sinf(val);
                float q0 = q->values[i * head_size + j];
                float q1 = q->values[i * head_size + j + 1];
                if (i < model.n_kv_heads) {
                    float k0 = k[i * head_size + j];
                    float k1 = k[i * head_size + j + 1];
                    k[i * head_size + j] = k0 * fcr - k1 * fci;
                    k[i * head_size + j + 1] = k0 * fci + k1 * fcr;
                }
            }
        } 

        // attention
        int h;
        #pragma omp parallel for private(h)
        for (h = 0; h < model.n_heads; h++) {
            float* q_head = q->values + h * head_size;
            for(int t = 0; t <= pos; t++) {
                int k_head_idx_mapped_to_h = h / (n_heads / n_kv_heads); // ref: notes - 1
                float* k_ht = k->values + t * kv_dim + k_head_idx_mapped_to_h * head_size;
                float score = 0.0f;
                for (int i = 0; i < head_size; i++) {
                    score += q_head[i] * k_ht[i];
                }
                score /= sqrtf(head_size);
                att[t] = score;
            }

            // softmax
            float max_val = att[0];
            for (int i = 1; i < pos+1; i++) {
                if (att[i] > max_val) {
                    max_val = att[i];
                }
            }
            float sum = 0.0f;
            for (int i = 0; i < pos+1; i++) {
                att[i] = expf(att[i] - max_val);
                sum += att[i];
            }
            for (int i = 0; i < x->size; i++) {
                att[i] /= sum;
            }

            // weighted sum of values, store back into x
            memset(x->values + h * head_size, 0, head_size * sizeof(float))  // dim can be thought of as (n_heads, head_size)
            for(int t = 0; t <= pos; t++) {
                int v_head_idx_mapped_to_h = h / (n_heads / n_kv_heads);
                v_ht = v->values + t * kv_dim + v_head_idx_mapped_to_h * head_size;
                for(int i = 0; i < head_size; i++) {
                    x->values + h * head_size + i = att[t] * v_ht[i];
                }
            }
        }

        Arr* x1 = matmul(x, wo);
        for (int i = 0; i < dim; i++) {
            x->values[i] += x1->values[i]; // residual
        }

        // rms norm before ffn
        Arr* x2 = rmsnorm(x, rms_bffn_weight[l]);

        // ffn: w2(silu(w1(x)) + w3(x))
        Arr* w1_out = matmul()

    }
}

/*
notes:
------

1. 32 query heads of size 128 mapped to 8 key heads of size 128.
query head 0-3 -> key head 0
query head 4-7 -> key head 1
etc. therefore:
query head h -> h / (n_heads/n_kv_heads) 

 */


int main() {
    return 0;
}
