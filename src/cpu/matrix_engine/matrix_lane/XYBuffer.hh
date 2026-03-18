//===================================================================================
// Authors: chenwanqi
// Date: 2024/1/16
//===================================================================================
// Description:
// 1. Number of ports is fixed, but active ports will depend on data size.
// 2. Default datatype is int8.
//===================================================================================
#ifndef __CPU_MATRIX_XYBUFFER_HH__
#define __CPU_MATRIX_XYBUFFER_HH__

#include "sim/ticked_object.hh"
#include "params/XYBuffer.hh"
// #include "cpu/matrix_engine/matrix_lane/Matrix_Lane.hh"
#include "base/statistics.hh"
#include "cpu/matrix_engine/matrix_lane/Compute_Unit.hh"
#include "cpu/matrix_engine/common/inst_class.hh"
#include "cpu/matrix_engine/matrix_lane/Matrix_Lane.hh"

#include <queue>
#include <bitset>
#include <cassert>
#include <cstdint>
#include <vector>

namespace gem5
{
struct XYBufferParams;
class XYBuffer : public TickedObject
{
public:
    XYBuffer(const XYBufferParams &params);
    ~XYBuffer();

    void set_matrixLanePtr(MatrixLane* _matrix_lane, uint64_t _cu_row_size, uint64_t _cu_column_size);
    void regStats() override;
    
    void startTicking(uint64_t matrix_column, uint64_t matrix_row); //will used by upper module
    void stopTicking(); //可以在evaluate里面关掉也可以在外面关掉
    void evaluate() override;
    // void update_delay();
    // void reset_delay();
    // void check_dataRemain();
    
    // return the data depth in this buffer.
    uint64_t get_unsigned_size();
    uint64_t get_signed_size();
    // two usage: send data to cu array; receive data from cu array
    // function reload here.
    void receive_data(uint8_t data, uint32_t portID); //default is int8
    void receive_data(int8_t data, uint32_t portID);
    // void receive_data(uint16_t data, uint32_t portID); //FIXME: add these functions later!
    // void receive_data(int16_t data, uint32_t portID);
    // void receive_data(uint32_t data, uint32_t portID);
    // void receive_data(int32_t data, uint32_t portID);

    /**
     * @description: send one row or one column to all the cu.
     */
    void send_data(); // pop data from the buffer and send into the cu array

    //status function
    bool isBusy();
    bool isFull();

// private:
public:
    //MatrixLane* matrix_lane; // FIXME: Maybe this should be removed later!
    // std::vector<ComputeUnit *> compute_units; // 2D array compute uint
    MatrixLane* matrix_lane; 
    // control signal or wire
    bool isSigned = false;
    uint64_t data_size;// used to decide how many the ports will be active.
    uint64_t column_size;// row and column
    uint64_t row_size;
    // uint8_t delay_index;
    
    //python configuration
    uint64_t buffer_depth;
    bool XY; // 1 means X buffer, 0 means Y buffer. 
    uint64_t data_width;
    uint64_t num_port; // equals with the number of the column/row of the cu array, input ports will equals output ports
    
    // struct Bits_def {
    //     std::bitset<data_width> bits;
    // };


public:
    statistics::Scalar read_from_XYBuffer;
    statistics::Scalar write_to_XYBuffer;
    statistics::Scalar XYBuffer_occupied;

private:
    //To simulate one cycle delay
    std::vector<bool> unsigned_writeRequest;
    std::vector<bool> signed_writeRequest;
    // uint8_t unsigned_data_temp(num_port)(buffer_depth);
    // int8_t signed_data_temp(num_port)(buffer_depth);
    std::vector<std::queue<uint8_t>> unsigned_data_temp;
    std::vector<std::queue<int8_t>> signed_data_temp;   
    //bool writeRequest;
    // bool readRequest;

    //wire
    uint64_t cu_row_size;
    uint64_t cu_column_size;

    // register
    std::vector<std::queue<uint8_t>> unsigned_buffer;
    std::vector<std::queue<int8_t>> signed_buffer;
    // std::vector<bool> delay_ctl;
    bool data_ready;
    bool busy; //equals !empty
    bool full;
    bool receiving_data; //Indicate if the XYBuffer need to receive the data, in case of the wrong state
};

}// namespace gem5
#endif