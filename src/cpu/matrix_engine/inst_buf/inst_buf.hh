/*
 * @Author: superboy
 * @Date: 2025-09-30 23:55:09
 * @LastEditTime: 2025-10-16 17:06:41
 * @LastEditors: superboy
 * @Description: 
 * @FilePath: /gem5-rvm/src/cpu/matrix_engine/inst_buf/inst_buf.hh
 * 
 */
#ifndef __CPU_MATRIX_IB_HH__
#define __CPU_MATRIX_IB_HH__

#include <cstdint>
#include <cassert>
#include <queue>

#include "params/InstructionBuffer.hh"
#include "sim/ticked_object.hh"
#include "cpu/matrix_engine/matrix_engine.hh"
#include "cpu/matrix_engine/matrix_engine_interface.hh"
#include "base/statistics.hh"
// #include "cpu/o3/dyn_inst.hh"
#include "cpu/o3/cpu.hh"
namespace gem5
{

struct InstructionBufferParams;
class InstructionBuffer : public TickedObject
{
public:
    InstructionBuffer(const InstructionBufferParams &params);
    ~InstructionBuffer();

    void startTicking();
    void stopTicking();
    bool isFull();
    bool isBusy();
    void set_engine_interface_ptr(MatrixEngine* _matrix_engine, MatrixEngineInterface* _matrix_interface);
    void set_cpu_ptr(gem5::o3::CPU* _o3cpu);
    void regStats() override;
    void evaluate() override;
    bool meminst_Empty() { return (meminst_num == 0) ? true : false; }
    bool push_inst(gem5::o3::DynInstPtr head_inst, uint64_t src1, uint64_t src2);
    bool pop_inst();
    uint16_t meminst_num = 0;
private:

    MatrixEngine *matrix_engine;
    MatrixEngineInterface *matrix_interface;
    gem5::o3::CPU* o3cpu;
    bool pop_req;
    bool Full;
    bool wrt_req;
    std::queue<gem5::o3::DynInstPtr> inst_buf;
    std::queue<uint64_t> rs1_buf;
    std::queue<uint64_t> rs2_buf;
    bool offload_minst = false;
    bool busy = false;
    uint32_t IB_depth;
    // uint16_t meminst_num = 0;
public:
    statistics::Scalar IB_Entry_Used;
    statistics::Scalar inst_read;
    statistics::Scalar inst_write;
    statistics::Scalar buffer_stall;
};
} //namespace gem5



#endif