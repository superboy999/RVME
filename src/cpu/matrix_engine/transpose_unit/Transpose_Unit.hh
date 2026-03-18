#ifndef __CPU_MATRIX_TRANSPOSEUNIT_HH__
#define __CPU_MATRIX_TRANSPOSEUNIT_HH__

#include "sim/ticked_object.hh"
#include "params/TransposeUnit.hh"
// #include "cpu/matrix_engine/matrix_lane/Matrix_Lane.hh"
#include "base/statistics.hh"
#include "cpu/matrix_engine/matrix_engine.hh"
#include "cpu/matrix_engine/matrix_lane/Compute_Unit.hh"
#include "cpu/matrix_engine/common/inst_class.hh"

#include <queue>
#include <bitset>
#include <cassert>
#include <cstdint>
#include <vector>

namespace gem5
{
struct TransposeUnitParams;
class TransposeUnit : public TickedObject
{
public:
    TransposeUnit(const TransposeUnitParams &params);
    ~TransposeUnit();

    void set_matrixEnginePtr(MatrixEngine* _matrix_engine);
    void regStats() override;
    
    void startTicking(uint64_t matrix_column, uint64_t matrix_row); //will used by upper module
    void stopTicking(); //可以在evaluate里面关掉也可以在外面关掉
    void evaluate() override;
    // void update_delay();
    // void reset_delay();
    // void check_dataRemain();
    
    // return the data depth in this buffer.
    // uint64_t get_unsigned_size();
    uint64_t get_signed_size();
    void set_phy_id(uint16_t matrixreg_id);
    // two usage: send data to cu array; receive data from cu array
    // function reload here.
    // void receive_data(uint8_t data, uint32_t portID); //default is int8
    void receive_data(uint32_t portID, int8_t data);
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
    // MatrixLane* matrix_lane; 
    // control signal or wire
    bool isSigned = true;
    uint16_t reg_id;
    // uint64_t data_size;// used to decide how many the ports will be active.
    uint64_t column_size;// row and column
    uint64_t row_size;
    uint64_t buffer_depth;
    
    //python configuration
    // uint64_t buffer_depth;
    // uint64_t data_width;
    uint64_t num_port; // equals with the number of the column/row of the cu array, input ports will equals output ports
    
    // struct Bits_def {
    //     std::bitset<data_width> bits;
    // };


public:
    statistics::Scalar read_from_TransposeUnit;
    statistics::Scalar write_to_TransposeUnit;
    statistics::Scalar TransposeUnit_occupied;

    MatrixEngine *matrix_engine;

private:
    //To simulate one cycle delay
    // std::vector<bool> unsigned_writeRequest;
    // std::vector<bool> signed_writeRequest;
    // uint8_t unsigned_data_temp(num_port)(buffer_depth);
    // int8_t signed_data_temp(num_port)(buffer_depth);
    // std::vector<std::queue<uint8_t>> unsigned_data_temp;
    std::queue<int8_t> signed_data_temp;   
    //bool writeRequest;
    // bool readRequest;

    uint64_t send_num;
    uint64_t recv_num;

    // register
    // std::vector<std::queue<uint8_t>> unsigned_buffer;
    std::vector<std::queue<int8_t>> signed_buffer;
    // std::vector<bool> delay_ctl;
    bool data_ready;
    bool busy; //equals !empty
    bool full;
    bool receiving_data; //Indicate if the TransposeUnit need to receive the data, in case of the wrong state
};

}// namespace gem5
#endif