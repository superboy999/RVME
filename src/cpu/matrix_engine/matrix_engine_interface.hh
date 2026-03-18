/*
 * @Author: superboy
 * @Date: 2024-08-22 02:49:37
 * @LastEditTime: 2025-10-23 15:26:48
 * @LastEditors: superboy
 * @Description: 
 * @FilePath: /SJTU-matrix-engine/src/cpu/matrix_engine/matrix_engine_interface.hh
 * 
 */
#ifndef __CPU_MATRIX_ENGINE_INTERFACE_HH__
#define __CPU_MATRIX_ENGINE_INTERFACE_HH__

#include <algorithm>
#include <functional>
#include <numeric>
#include <string>
#include <vector>
#include <cassert>
#include <cstdint>

#include "arch/riscv/insts/matrix_static_inst.hh"
#include "base/statistics.hh"
#include "base/types.hh"
#include "cpu/matrix_engine/matrix_engine.hh"
#include "cpu/matrix_engine/dispatcher/matrix_dispatcher.hh"
#include "cpu/matrix_engine/inst_buf/inst_buf.hh"
#include "params/MatrixEngineInterface.hh"
#include "sim/sim_object.hh"
#include "cpu/thread_context.hh"
#include "cpu/o3/cpu.hh"
// #include "cpu/o3/thread_context.hh"

namespace gem5
{
struct MatrixEngineInterfaceParams;

class MatrixEngine;

// namespace o3{
//     class ThreadContext;
// }

class MatrixEngineInterface : public SimObject
{
public:

    MatrixEngineInterface(const MatrixEngineInterfaceParams &params);
    ~MatrixEngineInterface();

    bool requestGrant(RiscvISA::RiscvMatrixInst* insn);
    bool requestGrant_withoutReg(RiscvISA::RiscvMatrixInst* insn);

    // This function interface reveal that the scalar core will send no more than two value to matrix engine.
    // This also represent that scalar core just do some micro-decode of the matrix instruction.
    // void sendCommand(RiscvISA::RiscvMatrixInst *minst, ThreadContext *tc, std::function<void()> done_callback);
    void sendCommand(RiscvISA::RiscvMatrixInst *minst, ThreadContext *tc);
    // FIXME:cwq I dont know if 64bit is great, maybe 32bit will be fine?
    // void configMatrix(RiscvISA::RiscvMatrixInst *minst, ThreadContext *tc, uint64_t src1, std::function<void()> done_callback);
    void configMatrix(RiscvISA::RiscvMatrixInst *minst, ThreadContext *tc, uint64_t src1_value);
    // void loadstoreMatrix(RiscvISA::RiscvMatrixInst *minst, ThreadContext *tc, uint64_t src1, uint64_t src2, std::function<void()> done_callback);
    void loadstoreMatrix(RiscvISA::RiscvMatrixInst *minst, ThreadContext *tc, uint64_t src1_value, uint64_t src2_value);
    void scalar2matrixDataMove(RiscvISA::RiscvMatrixInst *minst, ThreadContext *tc, uint64_t src1_value, uint64_t src2_value);
    void matrix2scalarDataMove(RiscvISA::RiscvMatrixInst *minst, ThreadContext *tc, uint64_t src1_value, std::function<void(uint64_t)> done_callback);
    void matrixDataMove(RiscvISA::RiscvMatrixInst *minst, ThreadContext *tc);
    void broadcastMatrix(RiscvISA::RiscvMatrixInst *minst, ThreadContext *tc, uint64_t imm3);
    void set_cpu_ptr(gem5::o3::CPU* _o3cpu);
    // void set_engine_interface_ptr(MatrixEngine* _matrix_engine, MatrixEngineInterface* _matrix_interface);
    void regStats() override;
    bool busy();
    bool isBlocked(); // once it has been blocked, none matrix mem inst can be offload from matrix engine, but for arith and config are ok.
    void setBlockStatus(bool status);
    void mzeroCmd(RiscvISA::RiscvMatrixInst *minst , ThreadContext *tc);
    void sendSpmCommand(RiscvISA::RiscvMatrixInst *minst, uint64_t src1_value, uint64_t src2_value, ThreadContext *tc);
    void sendSpmSync(RiscvISA::RiscvMatrixInst *minst, ThreadContext *tc);
// private:
public:
    MatrixEngine *matrix_engine;
    InstructionBuffer *inst_buffer;
    gem5::o3::CPU* o3cpu;
    // MatrixDispatcher *matrix_dispatcher;

    statistics::Scalar matrix_config_inst;
    statistics::Scalar matrix_arith_inst;
    statistics::Scalar matrix_mem_inst;
    statistics::Scalar matrix_mzero;
    statistics::Scalar matrix_spm_inst;
private:
    bool blocked = false;
};

} //namespace gem5

#endif