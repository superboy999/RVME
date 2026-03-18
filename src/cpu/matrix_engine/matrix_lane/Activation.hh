/*
 * @Author: superboy
 * @Date: 2024-06-17 14:55:49
 * @LastEditTime: 2024-06-17 15:51:41
 * @LastEditors: superboy
 * @Description: Designed as no delay module, totally combination circuit
 * @FilePath: /gem5-rvm/src/cpu/matrix_engine/matrix_lane/Activation.hh
 * 
 */
#ifndef __MATRIX_LANE_ACTIVATION_HH__
#define __MATRIX_LANE_ACTIVATION_HH__

#include <cstdint>

#include "sim/sim_object.hh"
#include "params/Activation.hh"

namespace gem5
{

struct ActivationParams;

class Activation : public SimObject
{
public:
    Activation(const ActivationParams &params);
    ~Activation();

    int32_t activate(int32_t data);
    uint32_t activate(uint32_t data);
private:
    //python configuration
    uint8_t mode; //0 means RELU

    // wire
    int32_t s_data;
    uint32_t u_data;
};

} //namespace gem5

#endif