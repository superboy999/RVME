#ifndef __CPU_MATRIX_ADDER_TREE_HH__
#define __CPU_MATRIX_ADDER_TREE_HH__

#include <string>
#include <vector>
#include <cassert>
#include <cstdint>
#include <queue>
#include <array>

#include "sim/ticked_object.hh"
#include "params/AdderTree.hh"
#include "base/statistics.hh"
#include "cpu/matrix_engine/matrix_lane/XYBuffer.hh"
#include "cpu/matrix_engine/matrix_lane/ZFPBuffer.hh"
#include "cpu/matrix_engine/matrix_lane/Activation.hh"
#include "cpu/matrix_engine/matrix_lane/Quantization.hh"
#include "cpu/o3/cpu.hh"

namespace gem5
{

struct AdderTreeParams;
class AdderTree : public TickedObject
{
public:

    AdderTree(const AdderTreeParams &params);
    ~AdderTree();

    void set_cpu_ptr(gem5::o3::CPU* _o3cpu);

    void regStats() override;

    void evaluate() override;
    bool outputdata_ready();
    
    float fp_mul(uint8_t x, uint8_t y);

    void startTicking(uint32_t partial_num);
    void stopTicking();

    // void unsigned_mac(uint8_t data_x, uint8_t data_y); //will not really compute
    // void signed_mac(int8_t data_x, int8_t data_y);
    // void get_unsigned_xdata(uint8_t xdata);
    // void get_unsigned_ydata(uint8_t ydata);
    void get_signed_xdata(int8_t xdata, uint32_t row_num);
    void get_signed_ydata(int8_t ydata, uint32_t column_num);

    void output_data(); //Need choose order in matrix lane

    bool computeDone();
    bool signed_ydata_ready();
    int8_t rd_xdata(uint32_t row_num);
    
    void set_zfpbufferPtr(ZFPBuffer* _zfpbuffer);
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

    ZFPBuffer* zfpbuffer;
    Quantization* quantization_layer;
    Activation* activation_layer;
    //=====register=======
    std::vector<std::vector<std::queue<float>>> signed_data;    

    uint8_t compute_pp_idx = 0;
    uint8_t output_pp_idx = 0;
    uint32_t u_pingpong[2] = {0, 0};
    int32_t s_pingpong[2] = {0, 0};
    // bool busy;
    
    //<xdata, ydata>
    std::queue<std::array<uint8_t, 2>> unsigned_data_queue;
    std::queue<std::array<int8_t, 2>> signed_data_queue;

    uint32_t compute_cnt;
    // uint32_t transfer_cnt; //FIX:0624, transfer to adjacent cu 
    bool compute_done;
    uint32_t output_cnt;
    bool output_done;

    //======control signal========
    // bool xdata_ready = false;
    // bool ydata_ready = false;
    std::vector<int> xdata_ready;
    bool ydata_ready;
    bool recv_xdata;
    bool recv_ydata;
    bool isSigned;
    uint32_t recv_ydata_cnt;
    uint32_t psum;
    uint8_t WIDEN;
    // bool readRequest = false; // cancel this, because readout is just one cycle.

    // ========wire=========
    uint8_t u_data_x; //only support int8 data type
    uint8_t u_data_y;
    // uint32_t u_data_temp;
    std::vector<std::vector<int8_t>> s_data_x;
    std::vector<int8_t> s_data_temp_x;
    std::vector<std::vector<int8_t>> s_data_y;
    std::vector<int8_t> s_data_temp_y;
    // int32_t s_data_temp;

    //python configuration
    bool en_activate;

    uint64_t num_port;

public:

    AdderTree* next_adder_tree;
    statistics::Scalar computeNumber;

    gem5::o3::CPU* o3cpu;

    uint32_t y_data_cnt;

    uint32_t computenum;
    
    uint64_t adder_tree_row_size;
    uint64_t adder_tree_column_size;

// private:
//     struct AdderTreeStat : public statistics::Group
//     {
//         AdderTreeStat(AdderTree* parent);
//         void regStats() override;

//         statistics::Scalar computeNumber;
//     }
};

}

#endif