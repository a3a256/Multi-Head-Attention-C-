#pragma once
#include <iostream>
#include <iomanip>
#include <vector>
#include <math.h>

std::vector<std::vector<float>> matmul(std::vector<std::vector<float>> one, std::vector<std::vector<float>> two){
    int i, j, k;
    std::vector<float> temp;
    std::vector<std::vector<float>> res;
    float _sum = 0.0f;
    for(i=0; i<one.size(); i++){
        for(j=0; j<two[0].size(); j++){
            _sum = 0.0f;
            for(k=0; k<one[0].size(); k++){
                _sum += one[i][k]*two[k][j];
            }
            temp.push_back(_sum);
        }
        res.push_back(temp);
        std::vector<float>().swap(temp);
    }
    return res;
}

std::vector<std::vector<float>> softmax(std::vector<std::vector<float>> &mat){
    float _sum;
    int i, j;
    for(i=0; i<mat.size(); i++){
        _sum = 0.0f;
        for(j=0; j<mat[i].size(); j++){
            _sum += std::exp(mat[i][j]);
        }
        for(j=0; j<mat[i].size(); j++){
            mat[i][j] = mat[i][j]/_sum;
        }
    }

    return mat;
}

std::vector<std::vector<float>> transpose(std::vector<std::vector<float>> mat){
    int i, j;
    std::vector<std::vector<float>> res(mat[0].size(), std::vector<float>(mat.size(), 0.0f));
    for(j=0; j<mat[0].size(); j++){
        for(i=0; i<mat.size(); i++){
            res[j][i] = mat[i][j];
        }
    }

    return res;
}