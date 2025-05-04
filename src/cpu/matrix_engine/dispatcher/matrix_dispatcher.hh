// ==========================================
// Author: cwq
// Last Date: 2024/2/28
// Description: Decode, and Will send to queue first, reorder and rename! Finally send to functional unit!
// ==========================================
#ifndef __CPU_MATRIX_DISPATCHER_HH__
#define __CPU_MATRIX_DISPATCHER_HH__

#include <cassert>
#include <cstdint>
#include <deque>

#include "arch/riscv/insts/matrix_static_inst.hh"
#include "sim/ticked_object.hh"
// #include "cpu/matrix_engine/reorder_buffer/reorder_buffer.hh"
// #include "cpu/matrix_engine/rename/matrix_rename.hh"
#include "cpu/matrix_engine/mmu/matrix_mmu.hh"
#include "cpu/matrix_engine/matrix_engine.hh"
#include "base/statistics.hh"
// #include "cpu/o3/thread_context.hh"
#include "cpu/thread_context.hh"
#include "cpu/matrix_engine/scoreboard/matrix_scoreboard.hh"
namespace gem5
{

struct MatrixDispatcherParams;

// namespace o3{
//     class ThreadContext;
// }

class MatrixDispatcher : public TickedObject
{
public:
    MatrixDispatcher(const MatrixDispatcherParams &params);
    ~MatrixDispatcher();

    void set_matrixEnginePtr(MatrixEngine* _matrix_engine);

    void renameMatrixInst(RiscvISA::RiscvMatrixInst &minst, ScoreBoard_Entry* matrix_sbe);
    void set_rob_entry(RiscvISA::RiscvMatrixInst &minst, ScoreBoard_Entry* matrix_sbe);
    void sendInstToCQ(ScoreBoard_Entry* matrix_sbe, uint64_t src1_value, ThreadContext *tc);
    void sendInstToMQ(ScoreBoard_Entry* matrix_sbe, uint64_t src1_value, uint64_t src2_value, ThreadContext *tc);
    void sendInstToAQ(ScoreBoard_Entry* matrix_sbe, ThreadContext *tc);

    bool dispatchGrant(RiscvISA::RiscvMatrixInst &minst); //对外的接口函数
    bool dispatchGrant_withoutReg(RiscvISA::RiscvMatrixInst &minst);
    // void dispatchInst();
    // bool issueGrant();

    void regStats() override;
    void evaluate() override;
    void startTicking();
    void stopTicking();
    bool isOccupied();
    class MemQueueEntry {
    public:
        MemQueueEntry(ScoreBoard_Entry &matrix_sbe, uint64_t src1_value, uint64_t src2_value, ThreadContext &_tc):
        matrix_sbe(matrix_sbe), start_addr(src1_value), row_stride(src2_value), issued(false), tc(_tc)
        {
            // tc = _tc;
        }
        ~MemQueueEntry(){};

        ScoreBoard_Entry &matrix_sbe;
        ThreadContext& tc;
        const uint64_t start_addr;
        const uint64_t row_stride;
        bool issued;
    };
    class ArithQueueEntry {
    public:
        ArithQueueEntry(ScoreBoard_Entry &matrix_sbe, uint64_t msrc1, uint64_t msrc2, uint16_t md, ThreadContext &_tc):
        matrix_sbe(matrix_sbe), ms1(msrc1), ms2(msrc2), md(md), issued(false), tc(_tc)
        {
            // tc = _tc;
        }
        ~ArithQueueEntry(){};

        ScoreBoard_Entry &matrix_sbe;
        ThreadContext& tc;
        const uint16_t ms1;
        const uint16_t ms2;
        const uint16_t md;
        bool issued;
    };
public:
    std::deque<MemQueueEntry* > Memory_Queue;
    std::deque<ArithQueueEntry* > Arithmetic_Queue;
    std::deque<ScoreBoard_Entry *> cfg_Queue;
    // MatrixRename *matrix_rename;
    // ReorderBuffer *matrix_reorder_buffer;
    // MatrixMMU* matrix_mmu;
    MatrixEngine *matrix_engine;

    statistics::Scalar WaitingForReg;  //counting the waiting cycle
    statistics::Scalar LaneWaitingForReg;  //counting the waiting cycle
    statistics::Scalar WaitingForLane;
    statistics::Scalar LaneisFree;
    statistics::Scalar hasFreeReg;
    statistics::Scalar allLaneisFree;
    statistics::Scalar MatrixMQentryUsed;
    statistics::Scalar MatrixAQentryUsed;
    statistics::Scalar AQread;
    statistics::Scalar AQwrite;
    statistics::Scalar MQread;
    statistics::Scalar MQwrite;

    statistics::Scalar testnum;
    statistics::Scalar matrix_load_inst;
    statistics::Scalar matrix_store_inst;
private:
    //python configuration;
    const uint32_t MQ_depth;
    const uint32_t AQ_depth;
    
    //control signal
    // bool config_stall; //Delete for cancel the config stall in 24/7/9
    bool OoO; 
    bool XY; //Indicate ma and mb, false means ma, true means mb.
    uint8_t transfer_num;
    // uint8_t sizeM;
    // uint8_t sizeN;
    // uint16_t sizeK;
    bool isBusy;
    //For one cycle delay
    // bool dispatchArithReq;
    // bool dispatchMemReq;

    bool issueArithReq;
    bool issueMemReq;

    // RiscvISA::RiscvMatrixInst minst_temp;
};
} //namespace gem5

#endif