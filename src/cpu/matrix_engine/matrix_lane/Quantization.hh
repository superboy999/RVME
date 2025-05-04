/*
 * @Author: superboy
 * @Date: 2024-06-17 16:16:35
 * @LastEditTime: 2024-06-19 15:05:36
 * @LastEditors: superboy
 * @Description: 
 * @FilePath: /gem5-rvm/src/cpu/matrix_engine/matrix_lane/Quantization.hh
 * 
 */

#ifndef __MATRIX_LANE_QUANTIZATION_HH__
#define __MATRIX_LANE_QUANTIZATION_HH__

#include <cstdint>
#include <cmath>

#include "sim/sim_object.hh"
#include "params/Quantization.hh"

namespace gem5
{

struct QuantizationParams;

class Quantization : public SimObject
{
public:
    Quantization(const QuantizationParams &params);
    ~Quantization();

    int8_t quantize(int32_t data);
    uint8_t quantize(uint32_t data);
private:
    //python configuration
    float scale_factor; 

    // wire
    int8_t s_data;
    uint8_t u_data;
};

} //namespace gem5

#endif