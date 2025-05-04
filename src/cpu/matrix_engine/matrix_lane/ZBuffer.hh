//===================================================================================
// Authors: chenwanqi
// Date: 2024/1/17
//===================================================================================
// Description:
// 1. Number of ports is fixed, but active ports will depend on data size.
// 2. Default datatype is int8.
//===================================================================================
#ifndef __CPU_MATRIX_ZBUFFER_HH__
#define __CPU_MATRIX_ZBUFFER_HH__

#include "sim/ticked_object.hh"
#include "params/ZBuffer.hh"
#include "base/statistics.hh"
#include "cpu/matrix_engine/matrix_lane/Compute_Unit.hh"
#include "cpu/matrix_engine/common/inst_class.hh"
#include "cpu/matrix_engine/matrix_lane/Matrix_Lane.hh"
#include "cpu/matrix_engine/matrix_engine.hh"

#include <queue>
#include <deque>
#include <array>
#include <bitset>
#include <cassert>
#include <cstdint>
#include <vector>

namespace gem5
{
struct ZBufferParams;
class ZBuffer : public TickedObject
{
public:
    ZBuffer(const ZBufferParams &params);
    ~ZBuffer();

    void set_topPtr(MatrixLane* _matrix_lane, MatrixEngine* _matrix_engine);
    void regStats() override;
    
    void startTicking(uint64_t matrix_column, uint64_t matrix_row, uint8_t idx); //will used by upper module
    void stopTicking(); //可以在evaluate里面关掉也可以在外面关掉
    void evaluate() override;

    // return the data depth in this buffer.
    uint64_t get_unsigned_size();
    uint64_t get_signed_size();

    // void receive_data(uint8_t data, uint32_t coordinate_x, uint32_t coordinate_y, uint8_t offset); //default is int8
    // void receive_data(int8_t data, uint32_t coordinate_x, uint32_t coordinate_y, uint8_t offset);

    void receive_data(uint8_t data, uint32_t coordinate_x, uint32_t coordinate_y); //default is int8
    void receive_data(int8_t data, uint32_t coordinate_x, uint32_t coordinate_y);

    void send_data(uint8_t dstReg_idx); // FIXME: send to mmu? answer is no!
    void send_req();

    //status function
    bool isBusy();
    bool isFull();

private:

    MatrixLane* matrix_lane;
    MatrixEngine* matrix_engine;
    // control signal or wire
    bool isSigned = false;
    uint64_t data_size;// used to decide how many the ports will be active.同样也是这个buffer在这个matrix尺寸下，该容纳的数据量
    uint64_t column_size;// row and column
    uint64_t row_size;
    
    //python configuration
    uint64_t buffer_depth;
    uint64_t data_width;
    uint64_t num_port; // equals with the number of the column/row of the cu array, input ports will equals output ports

public:
    statistics::Scalar read_from_ZBuffer;
    statistics::Scalar write_to_ZBuffer;
    statistics::Scalar ZBuffer_occupied;

private:
    //To simulate one cycle delay
    std::vector<bool> unsigned_writeRequest;
    std::vector<bool> signed_writeRequest;

    // std::vector<std::deque<std::array<uint8_t, 4>>> unsigned_data_temp;
    // std::vector<std::deque<std::array<int8_t, 4>>> signed_data_temp;   
    std::vector<std::deque<uint8_t>> unsigned_data_temp;
    std::vector<std::deque<int8_t>> signed_data_temp;   

    // register
    // std::vector<std::deque<std::array<uint8_t, 4>>> unsigned_buffer;
    // std::vector<std::deque<std::array<int8_t, 4>>> signed_buffer;
    std::vector<std::deque<uint8_t>> unsigned_buffer;
    std::vector<std::deque<int8_t>> signed_buffer;
    bool data_ready;
    bool busy; //equals !empty
    bool full;
    bool sendReq;
    uint8_t send_cnt;
    uint8_t dstReg_idx;
};

}// namespace gem5
#endif