/*
 * @Author: superboy
 * @Date: 2025-10-04 21:12:53
 * @LastEditTime: 2025-10-19 01:56:24
 * @LastEditors: superboy
 * @Description: 
 * @FilePath: /gem5-rvm/src/cpu/matrix_engine/elementwise_unit/ew_unit.hh
 * 
 */

#ifndef __CPU_MATRIX_EW_UNIT_HH__
#define __CPU_MATRIX_EW_UNIT_HH__

#include <cstdint>
#include <cassert>
#include <array>
#include <vector>
#include "params/ElementwiseUnit.hh"
#include "sim/ticked_object.hh"
#include "cpu/matrix_engine/matrix_engine.hh"
#include "base/statistics.hh"
#include "cpu/matrix_engine/common/inst_class.hh"
#include "cpu/matrix_engine/scoreboard/matrix_scoreboard.hh"
namespace gem5
{
struct ElementwiseUnitParams;
class ElementwiseUnit : public TickedObject
{
public:
    ElementwiseUnit(const ElementwiseUnitParams &params);
    ~ElementwiseUnit();

    void startTicking();
    void stopTicking();
    void acc_req(uint8_t _dest, uint8_t _sizem, uint8_t _idx, bool _isSigned); //send from zbuffer
    void accu_one_row(std::array<uint8_t, 8> row_data, uint8_t idx); //send from zbuffer
    bool isOccupied(uint8_t idx); //通过这个判断是否能接受指令
    void set_matrixEnginePtr(MatrixEngine* _matrix_engine);
    std::array<int32_t, 8> elementwise_operation_s(std::array<int32_t, 8> A, std::array<int32_t, 8> B, MatrixInstClass::EW_OP ew_op);
    std::array<uint32_t, 8> elementwise_operation_u(std::array<uint32_t, 8> A, std::array<uint32_t, 8> B, MatrixInstClass::EW_OP ew_op);
    void regStats() override;
    void evaluate() override;
    bool isIdle();
    bool isDone(uint8_t idx) { return done[idx]; }
    void resetDone(uint8_t idx) { done[idx] = false; }
    void recv_opera(ScoreBoard_Entry& matrix_sbe, uint8_t _idx, bool _isSigned); //send from ArithQueue

private:
    // bool busy = false;
    std::vector<bool> busy;
    // MatrixInstClass::EW_OP cur_ew_op;
    std::vector<MatrixInstClass::EW_OP> cur_ew_op;
    MatrixEngine* matrix_engine;
    // uint8_t exe_num = 0; //now exe times
    std::vector<uint8_t> exe_num;
    // uint8_t sizen = 0; // aim to exe times
    std::vector<uint8_t> sizem;
    // uint8_t delay_cycle = 0; // delay cycles
    std::vector<uint8_t> delay_cycle;
    // uint8_t src1, src2, dest; //record the operands
    std::vector<uint8_t> src1, src2, dest; //record the operands
    // bool acc_req = false; //whether there is an accumulate request
    std::vector<bool> accReq;
    // bool opera_req = false; //whether there is an operation request
    std::vector<bool> operaReq;

    std::vector<bool> isExe;

    std::vector<bool> done;

    std::vector<bool> isSigned;
    // struct OP_delay{
    //     uint8_t MADD_delay = 1;
    //     uint8_t MSUB_delay = 1;
    //     uint8_t MMUL_delay = 1;
    //     uint8_t MMAX_delay = 1;
    //     uint8_t MMIN_delay = 1;
    //     uint8_t MSHIFTR_delay = 1;
    //     uint8_t MSHIFTL_delay = 1;
    // };
    uint8_t OP_delay[7] = {1, 1, 1, 1, 1, 1, 1}; //delay cycles for different operations

    uint8_t parallel_ewu; //number of parallel ewu
public:
    statistics::Scalar ew_op_accepted;
    statistics::Scalar acc_executed;
};
} //namespace gem5

#endif