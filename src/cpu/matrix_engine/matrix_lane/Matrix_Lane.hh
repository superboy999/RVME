#ifndef __CPU_MATRIX_LANE_HH__
#define __CPU_MATRIX_LANE_HH__

// #include <queue>
#include <vector>
#include <functional>
#include <cassert>
#include <cstdint>

#include "sim/ticked_object.hh"
#include "params/MatrixLane.hh"
#include "base/statistics.hh"
#include "cpu/matrix_engine/matrix_lane/XYBuffer.hh"
#include "cpu/matrix_engine/matrix_lane/ZBuffer.hh"
#include "cpu/matrix_engine/common/inst_class.hh"
#include "cpu/matrix_engine/matrix_lane/Compute_Unit.hh"

#include "cpu/matrix_engine/matrix_registerfile/matrix_reg.hh"
#include "cpu/matrix_engine/scoreboard/matrix_scoreboard.hh"
#include "cpu/matrix_engine/matrix_engine.hh"
namespace gem5
{

struct MatrixLaneParams;
class MatrixLane : public TickedObject
{
public:
    MatrixLane(const MatrixLaneParams &params);
    ~MatrixLane();

    // void issue_inst(MatrixInstClass::InstType inst_type, MatrixInstClass::DataType data_type, MatrixInstClass::MatrixSize matrix_size, bool signed); //just transfer inst
    void issue_inst(ScoreBoard_Entry* matrix_sbe); //just transfer inst

    //transfer data to xbuffer, but one column per operation.如要传完整个矩阵，需要调用多次该函数。
    void rec_x_matrix(uint8_t PhysRegIdx, uint8_t bank_idx, uint8_t entry_idx); 
    //transfer data one row per operation
    void rec_y_matrix(uint8_t PhysRegIdx, uint8_t bank_idx, uint8_t entry_idx); 

    // void out_matrix(); // FIXME: output Bandwidth has not decided!

    void startTicking(); // Let the lower level module start first, to realize the delay correctly!

    void stopTicking();

    void evaluate() override; //when done->1 and busy ->0, then stop()!
    void regStats() override;

    //status record
    // bool isDone();
    bool isOccupied();

    void set_matrixEnginePtr(MatrixEngine* _matrix_engine);
    // void recvInst (ScoreBoard_Entry* matrix_sbe);

private:

    // register or state machine or counter
    bool computeDone; // true when all partial sum is computed!
    // bool computing; //ADD for the compute unit connection
    bool saveZDone;
    bool sendtoZ;
    bool start_transZ;
    bool start_output;
    bool transOutDone;
    //bool transInDone; //from outside into the X/YBuffer
    bool busy; // true when this lane is executing an instruction!
    uint32_t toZcnt;

    bool occupy_x;
    bool occupy_y;
    // uint32_t toCUcnt;

    //python configuration
    // uint64_t inst_queue_depth; //Two defined queue upon this line, will be sized to this value!
    uint64_t xbuffer_ports_num;
    uint64_t ybuffer_ports_num;
    uint64_t cu_row_size;
    uint64_t cu_column_size;
    // uint8_t WIDEN; //moved to public
    //control signal
    // MatrixInstClass::MatrixSize matrix_size;
public:
    //pointer
    XYBuffer *XBuffer;
    XYBuffer *YBuffer;
    ZBuffer *zbuffer;
    std::vector<ComputeUnit *> compute_units; // 2D array compute uint 
    MatrixInstClass* lane_minst;// every lane can only receive one inst. Also control signal!
    MatrixInstClass tranz_minst;
    // MatrixRF* matrix_reg;//FIXME: Add this later!
    MatrixEngine* matrix_engine;
    // ScoreBoard_Entry* matrix_sbe;
private:
    //To simulate one cycle delay
    std::vector<std::queue<uint8_t>> unsigned_xdata_temp;
    std::vector<std::queue<int8_t>> signed_xdata_temp;
    std::vector<std::queue<uint8_t>> unsigned_ydata_temp;
    std::vector<std::queue<int8_t>> signed_ydata_temp;
    bool writeXRequest;
    bool writeYRequest;
    bool issue_req;

public:
    statistics::Scalar InstNum;
    bool InstDone;
    uint8_t WIDEN;
};
} // namespace gem5

#endif