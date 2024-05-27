#pragma once
#include <iostream>
#include <iomanip>
#include <vector>
#include <math.h>
#include "linalg_ops.h"
#include <random>
#include <climits>
#include <numeric>

class Linear{
    public:
        int in_features, out_features;
        std::vector<std::vector<float>> parameters;
        Linear(int in_channels, int out_channels){
            in_features = in_channels;
            out_features = out_channels;

            int i, j;
            std::vector<float> temp;
            float val;
            for(i=0; i<in_channels; i++){
                for(j=0; j<out_channels; j++){
                    val = random_value();
                    temp.push_back(val);
                }
                parameters.push_back(temp);
                std::vector<float>().swap(temp);
            }
        }

        std::vector<std::vector<float>> forward(std::vector<std::vector<float>> x){
            return matmul(x, parameters);
        }
};

class MultiHeadAttention{
    public:
        int em_size;
        int heads;
        std::vector<Linear> q_weights;
        std::vector<Linear> k_weights;
        std::vector<Linear> v_weights;

        std::vector<Linear> attention_fc;

        MultiHeadAttention(int em_shape, int num_heads){
            em_size = em_shape;
            heads = num_heads;
            int i;
            for(i=0; i<heads; i++){
                q_weights.push_back(Linear(em_shape, em_shape));
                k_weights.push_back(Linear(em_shape, em_shape));
                v_weights.push_back(Linear(em_shape, em_shape));
            }
            Linear layer(em_size*heads, em_size);
            attention_fc.push_back(layer);
        }

        std::vector<std::vector<float>> forward(std::vector<std::vector<float>> q, std::vector<std::vector<float>> k, std::vector<std::vector<float>> v){
            std::vector<std::vector<std::vector<float>>> outputs;
            std::vector<std::vector<float>> attention;
            int i, j;
            for(i=0; i<heads; i++){
                attention = attention_layer(q_weights[i].forward(q),
                                            k_weights[i].forward(k),
                                            v_weights[i].forward(v),
                                            em_size);
                outputs.push_back(attention);
            }
            std::vector<std::vector<float>> result;
            result = outputs[0];
            for(i=1; i<outputs.size(); i++){
                for(j=0; j<outputs[i].size(); j++){
                    result[j].insert(result[j].end(), outputs[i][j].begin(), outputs[i][j].end());
                }
            }
            return attention_fc[0].forward(result);
        }

    private:

        std::vector<std::vector<float>> attention_layer(std::vector<std::vector<float>> q,
                                                std::vector<std::vector<float>> k,
                                                std::vector<std::vector<float>> v,
                                                int dk){
            std::vector<std::vector<float>> first_mul;
            first_mul = matmul(q, transpose(k));
            float dk_sqrt = std::sqrt((float)dk);
            int i, j;
            for(i=0; i<first_mul.size(); i++){
                for(j=0; j<first_mul[i].size(); j++){
                    first_mul[i][j] = first_mul[i][j]/dk_sqrt;
                }
            }
            first_mul = softmax(first_mul);
            first_mul = matmul(first_mul, v);

            return first_mul;
        }
};