#pragma once
#include <iostream>
#include <iomanip>
#include <vector>
#include <math.h>
#include "linalg_ops.h"
#include <random>
#include <climits>
#include <numeric>

class MultiHeadAttention{
    public:
        int em_size;
        int heads;
        std::vector<std::vector<std::vector<float>>> q_weights;
        std::vector<std::vector<std::vector<float>>> k_weights;
        std::vector<std::vector<std::vector<float>>> v_weights;
        MultiHeadAttention(int em_shape, int num_heads){
            em_size = em_shape;
            heads = num_heads;
            int i, j, k;
            for(i=0; i<heads; i++){
                std::vector<std::vector<float>> matrix(heads, std::vector<float>(heads, 0.0f));
                for(j=0; j<heads; j++){
                    for(k=0; k<heads; k++){
                        matrix[j][k] = random_value();
                    }
                }
                q_weights.push_back(matrix);
                std::vector<std::vector<float>>().swap(matrix);
            }

            for(i=0; i<heads; i++){
                std::vector<std::vector<float>> matrix(heads, std::vector<float>(heads, 0.0f));
                for(j=0; j<heads; j++){
                    for(k=0; k<heads; k++){
                        matrix[j][k] = random_value();
                    }
                }
                k_weights.push_back(matrix);
                std::vector<std::vector<float>>().swap(matrix);
            }

            for(i=0; i<heads; i++){
                std::vector<std::vector<float>> matrix(heads, std::vector<float>(heads, 0.0f));
                for(j=0; j<heads; j++){
                    for(k=0; k<heads; k++){
                        matrix[j][k] = random_value();
                    }
                }
                v_weights.push_back(matrix);
                std::vector<std::vector<float>>().swap(matrix);
            }
        }

        std::vector<std::vector<float>> forward(std::vector<std::vector<float>> q, std::vector<std::vector<float>> k, std::vector<std::vector<float>> v){
            std::vector<std::vector<std::vector<float>>> outputs;
            std::vector<std::vector<float>> attention;
            int i;
            for(i=0; i<heads; i++){
                attention = attention_layer(matmul(q, q_weights[i]),
                                                matmul(k, k_weights[i]),
                                                matmul(v, v_weights[i]),
                                                em_size);
                outputs.push_back(attention);
            }
            std::vector<std::vector<float>> result;
            result = outputs[0];
            for(i=1; i<outputs.size(); i++){
                result.insert(result.end(), outputs[i].begin(), outputs[i].end());
            }
            return result;
        }

    private:

        float random_value(){
            std::random_device seeder;
            std::mt19937 rng(seeder());
            std::uniform_int_distribution<long> gen(INT_MIN, INT_MAX);
            return (float)gen(rng)/(float)RAND_MAX;
        }

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