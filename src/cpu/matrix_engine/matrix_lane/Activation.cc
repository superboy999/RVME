/*
 * @Author: superboy
 * @Date: 2024-06-17 14:55:33
 * @LastEditTime: 2024-06-19 14:26:33
 * @LastEditors: superboy
 * @Description: 
 * @FilePath: /gem5-rvm/src/cpu/matrix_engine/matrix_lane/Activation.cc
 * 
 */
#include "cpu/matrix_engine/matrix_lane/Activation.hh"
#include "debug/Activation.hh"

namespace gem5
{
Activation::Activation(const ActivationParams &params) :
    SimObject(params), mode(params.mode)
    {}

Activation::~Activation()
{}

int32_t Activation::activate(int32_t data)
{
    if(mode == 0){
        //RELU function body
        if( data > 0){
            s_data = data;
        } else {
            s_data = 0;
        }
    }
    return s_data;
}
uint32_t Activation::activate(uint32_t data)
{
    if(mode == 0){
        //RELU function body
        if( data > 0){
            u_data = data;
        } else {
            u_data = 0;
        }
    }
    return u_data;
}
}