/*
 * @Author: superboy
 * @Date: 2024-06-16 10:50:22
 * @LastEditTime: 2025-03-18 02:24:21
 * @LastEditors: superboy
 * @Description: 
 * @FilePath: /gem5-rvm/src/cpu/matrix_engine/matrix_lane/Compute_Unit.hh
 * 
 */
#ifndef __CPU_MATRIX_COMPUTE_UNIT_HH__
#define __CPU_MATRIX_COMPUTE_UNIT_HH__

#include <string>
#include <vector>
#include <cassert>
#include <cstdint>
#include <queue>
#include <array>

#include "sim/ticked_object.hh"
#include "params/ComputeUnit.hh"
#include "base/statistics.hh"
#include "cpu/matrix_engine/matrix_lane/XYBuffer.hh"
#include "cpu/matrix_engine/matrix_lane/ZBuffer.hh"
#include "cpu/matrix_engine/matrix_lane/Activation.hh"
#include "cpu/matrix_engine/matrix_lane/Quantization.hh"
namespace gem5
{

struct ComputeUnitParams;
class ComputeUnit : public TickedObject
{
public:
    ComputeUnit(const ComputeUnitParams &params);
    ~ComputeUnit();

    void regStats() override;

    void evaluate() override;
    void startTicking(uint32_t partial_num);
    void stopTicking();

    // void unsigned_mac(uint8_t data_x, uint8_t data_y); //will not really compute
    // void signed_mac(int8_t data_x, int8_t data_y);
    void get_unsigned_xdata(uint8_t xdata);
    void get_unsigned_ydata(uint8_t ydata);
    void get_signed_xdata(int8_t xdata);
    void get_signed_ydata(int8_t ydata);

    void output_data(); //Need choose order in matrix lane

    bool computeDone();
    void set_zbufferPtr(ZBuffer* _zbuffer);
    // bool isBusy();

    // change to systolic type connection
    // struct data_pair{
    //     uint8_t u_xdata;
    //     uint8_t u_ydata;

    //     int8_t s_xdata;
    //     int8_t s_ydata;
    // };
    // data_pair data_pair1;
    // data_pair transfer();
private:

    ZBuffer* zbuffer;
    Quantization* quantization_layer;
    Activation* activation_layer;
    //=====register=======
    uint32_t unsigned_data = 0;
    int32_t signed_data = 0;
    uint32_t activate_unsigned_data = 0;
    int32_t activate_signed_data = 0;
    

    uint8_t compute_pp_idx = 0;
    uint8_t output_pp_idx = 0;
    uint32_t u_pingpong[2] = {0, 0};
    int32_t s_pingpong[2] = {0, 0};
    // bool busy;
    
    //<xdata, ydata>
    std::queue<std::array<uint8_t, 2>> unsigned_data_queue;
    std::queue<std::array<int8_t, 2>> signed_data_queue;

    //quantized data
    uint8_t q_unsigned_data = 0;
    int8_t q_signed_data = 0;

    uint32_t coordinate[2];
    uint32_t compute_cnt;
    // uint32_t transfer_cnt; //FIX:0624, transfer to adjacent cu 
    bool compute_done;
    uint32_t output_cnt;
    bool output_done;

    //======control signal========
    // bool xdata_ready = false;
    // bool ydata_ready = false;    
    bool xdata_ready;
    bool ydata_ready;
    bool isSigned;
    uint32_t psum;
    uint8_t WIDEN;
    // bool readRequest = false; // cancel this, because readout is just one cycle.

    // ========wire=========
    uint8_t u_data_x; //only support int8 data type
    uint8_t u_data_y;
    // uint32_t u_data_temp;
    int8_t s_data_x;
    int8_t s_data_y;
    // int32_t s_data_temp;

    //python configuration
    bool en_activate;

public:
    statistics::Scalar computeNumber;

// private:
//     struct ComputeUnitStat : public statistics::Group
//     {
//         ComputeUnitStat(ComputeUnit* parent);
//         void regStats() override;

//         statistics::Scalar computeNumber;
//     }
};

}

#endif