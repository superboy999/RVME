// #include "cpu/matrix_engine/matrix_engine_interface.hh"

#include <string>
#include <vector>

#include "base/logging.hh"
#include "base/types.hh"
#include "sim/faults.hh"
#include "sim/sim_object.hh"
#include "debug/MatrixEngineInterface.hh"
#include "cpu/matrix_engine/matrix_engine_interface.hh"
#include <cassert>
#include <cstdint>

namespace gem5
{

MatrixEngineInterface::MatrixEngineInterface(const MatrixEngineInterfaceParams &params) : SimObject(params), matrix_engine(params.matrix_engine)
{
    matrix_engine->set_cpu_ptr(o3cpu);
}

MatrixEngineInterface::~MatrixEngineInterface(){}

void MatrixEngineInterface::regStats()
{
    SimObject::regStats();
    matrix_config_inst
        .name(name() + ".numMatrixConfigInst")
        .desc("Count configuration instructions");
    matrix_arith_inst
        .name(name() + ".numMatrixArithInst")
        .desc("Count Arithmatic Instructions");
    matrix_mem_inst
        .name(name() + ".numMatrixMemInst")
        .desc("Count Memory Instructions");
}

bool MatrixEngineInterface::requestGrant(RiscvISA::RiscvMatrixInst* insn)
{
    //FIXME:这个貌似要改成dispatchGrant的那个函数
    bool grant = matrix_engine->matrix_dispatcher->dispatchGrant(*insn);
    DPRINTF(MatrixEngineInterface, " Requesting a grant with answer: %d\n", grant);
    return grant;
}

bool MatrixEngineInterface::requestGrant_withoutReg(RiscvISA::RiscvMatrixInst* insn)
{
    //FIXME:这个貌似要改成dispatchGrant的那个函数
    bool grant = matrix_engine->matrix_dispatcher->dispatchGrant_withoutReg(*insn);
    DPRINTF(MatrixEngineInterface, " Requesting a grant with answer: %d\n", grant);
    return grant;
}

void MatrixEngineInterface::sendCommand(RiscvISA::RiscvMatrixInst *minst, ThreadContext *tc)
{
    DPRINTF(MatrixEngineInterface, "Sending a Compute command to the matrix engine\n");
    ScoreBoard_Entry* sbe = new ScoreBoard_Entry();
    if(!matrix_engine->matrix_dispatcher->isOccupied()){
        matrix_engine->matrix_dispatcher->startTicking();
    }
    matrix_engine->matrix_dispatcher->renameMatrixInst(*minst, sbe);
    matrix_engine->matrix_dispatcher->set_rob_entry(*minst, sbe);
    sbe->set_MatrixStaticInst(minst);
    matrix_engine->matrix_dispatcher->sendInstToAQ(sbe, tc);
    matrix_arith_inst++;
}

void MatrixEngineInterface::configMatrix(RiscvISA::RiscvMatrixInst *minst, ThreadContext *tc, uint64_t src1_value)
{
    DPRINTF(MatrixEngineInterface, "Sending a Config command to the matrix engine\n");
    DPRINTF(MatrixEngineInterface, "cfg value = %d\n", src1_value);
    ScoreBoard_Entry* sbe = new ScoreBoard_Entry();
    if(!matrix_engine->matrix_dispatcher->isOccupied()){
        matrix_engine->matrix_dispatcher->startTicking();
    }
    sbe->set_MatrixStaticInst(minst);
    sbe->set_rs1_value(src1_value);
    matrix_engine->matrix_dispatcher->sendInstToCQ(sbe, src1_value, tc);
    matrix_config_inst++;
}

void MatrixEngineInterface::loadstoreMatrix(RiscvISA::RiscvMatrixInst *minst, ThreadContext *tc, uint64_t src1_value, uint64_t src2_value)
{
    DPRINTF(MatrixEngineInterface, "Sending a LoadStore command to the matrix engine\n");
    ScoreBoard_Entry* sbe = new ScoreBoard_Entry();
    DPRINTF(MatrixEngineInterface, "src1_value = %d\n", src1_value);
    DPRINTF(MatrixEngineInterface, "src2_value = %d\n", src2_value);
    if(!matrix_engine->matrix_dispatcher->isOccupied()){
        matrix_engine->matrix_dispatcher->startTicking();
    }
    matrix_engine->matrix_dispatcher->renameMatrixInst(*minst, sbe);
    matrix_engine->matrix_dispatcher->set_rob_entry(*minst, sbe);
    sbe->set_MatrixStaticInst(minst);
    sbe->set_rs1_value(src1_value);
    sbe->set_rs2_value(src2_value);
    matrix_engine->matrix_dispatcher->sendInstToMQ(sbe, src1_value, src2_value, tc);
    matrix_mem_inst++;
}

bool MatrixEngineInterface::busy()
{
    bool busy = matrix_engine->isOccupied();
    return busy;
}

void MatrixEngineInterface::set_cpu_ptr(gem5::o3::CPU* _o3cpu)
{
    o3cpu = _o3cpu;
    matrix_engine->set_cpu_ptr(o3cpu);
}

// FIXMEcwq: maybe it should be integrated into python script
// MatrixEngineInterface *
// MatrixEngineInterfaceParams::create()
// {
//     return new MatrixEngineInterface(this);
// }

}
