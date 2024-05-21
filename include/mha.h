#pragma once
#include <iostream>
#include <iomanip>
#include <vector>
#include <math.h>
#include "linalg_ops.h"

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
}