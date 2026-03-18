/*
 * @Author: superboy
 * @Date: 2024-06-17 16:16:27
 * @LastEditTime: 2024-06-19 15:08:47
 * @LastEditors: superboy
 * @Description: 
 * @FilePath: /gem5-rvm/src/cpu/matrix_engine/matrix_lane/Quantization.cc
 * 
 */

#include "cpu/matrix_engine/matrix_lane/Quantization.hh"
#include "debug/Quantization.hh"

namespace gem5
{
Quantization::Quantization(const QuantizationParams &params) : SimObject(params), scale_factor(params.scale_factor)
{}

Quantization::~Quantization()
{}

int8_t Quantization::quantize(int32_t data)
{
    float scaledNumber = data * scale_factor;
    int32_t roundedNumber = std::round(scaledNumber);
    return static_cast<int8_t>(roundedNumber);
}

uint8_t Quantization::quantize(uint32_t data)
{
    float scaledNumber = data * scale_factor;
    int32_t roundedNumber = std::round(scaledNumber);
    return static_cast<uint8_t>(roundedNumber);
}

    
}