/*
 * @Author: superboy
 * @Date: 2025-09-30 23:55:00
 * @LastEditTime: 2025-10-18 14:38:21
 * @LastEditors: superboy
 * @Description: 
 * @FilePath: /gem5-rvm/src/cpu/matrix_engine/inst_buf/inst_buf.cc
 * 
 */

#include "cpu/matrix_engine/inst_buf/inst_buf.hh"
#include "cpu/o3/dyn_inst.hh"
#include "debug/InstructionBuffer.hh"
#include "arch/riscv/insts/matrix_static_inst.hh"

#include <cassert>
#include <cstdint>

namespace gem5
{
InstructionBuffer::InstructionBuffer(const InstructionBufferParams &params) :
    TickedObject(params), IB_depth(params.IB_depth)
{
    pop_req = false;
    Full = false;
    wrt_req = false;
    // for(uint32_t i = 0; i < IB_depth; i ++)
    // {
    //     inst_buf.push_back(new RiscvISA::RiscvMatrixInst());
    // }
}

InstructionBuffer::~InstructionBuffer(){}

void InstructionBuffer::set_engine_interface_ptr(MatrixEngine* _matrix_engine, MatrixEngineInterface* _matrix_interface)
{
    matrix_engine = _matrix_engine;
    matrix_interface = _matrix_interface;
}

void InstructionBuffer::startTicking()
{
    busy = true;
    DPRINTF(InstructionBuffer, "Matrix Engine Instruction Buffer is start working!\n");
    start();
}

void InstructionBuffer::stopTicking()
{
    DPRINTF(InstructionBuffer, "Matrix Engine Instruction Buffer is stop working!\n");
    busy = false;
    stop();
}

bool InstructionBuffer::isFull()
{
    Full = (inst_buf.size() >= IB_depth) ? true : false;
    return Full;
}

bool InstructionBuffer::isBusy()
{
    return busy;
}

void InstructionBuffer::regStats()
{
    TickedObject::regStats();

    IB_Entry_Used
    .name(name() + ".IB_entry_use")
    .desc("Number of Instruction Buffer entry used!");
    inst_read
    .name(name() + ".inst_read")
    .desc("Number of Instruction Buffer entry read!");
    inst_write
    .name(name() + ".inst_write")
    .desc("Number of Instruction Buffer entry written!");

}

bool InstructionBuffer::push_inst(gem5::o3::DynInstPtr head_inst)
{
    // assert(!isFull());
    if(isFull()){
        return false;
    }
    inst_buf.push(head_inst);
    wrt_req = true;
    inst_write++;
    DPRINTF(InstructionBuffer, "Push instruction into IB, current size: %d\n", inst_buf.size());
    return true;
}

bool InstructionBuffer::pop_inst()
{
    if(inst_buf.empty()){
        return false;
    }
    // assert(!inst_buf.empty());
    inst_buf.pop();
    pop_req = false;
    inst_read++;
    DPRINTF(InstructionBuffer, "Pop instruction from IB, current size: %d\n", inst_buf.size());
    return true;
}

void InstructionBuffer::evaluate()
{
    offload_minst = false;
    if((double)inst_buf.size() > IB_Entry_Used.value())
        IB_Entry_Used = inst_buf.size();
    if (!inst_buf.empty()){
        gem5::o3::DynInstPtr head_inst = inst_buf.front();
        RiscvISA::RiscvMatrixInst *matrix_inst = dynamic_cast<RiscvISA::RiscvMatrixInst*>(head_inst->staticInst.get());
        if(matrix_inst->isStore()&&matrix_interface->requestGrant_withoutReg(matrix_inst)||(!matrix_inst->isStore()&&matrix_interface->requestGrant(matrix_inst))){
            DPRINTF(InstructionBuffer, "IB offload a matrix instruction to matrix engine!\n");
            uint64_t src1, src2;
            uint64_t imm;
            if(matrix_inst->isLoad()||matrix_inst->isStore()){
                src1 = head_inst->getRegOperand(matrix_inst, 0);
                src2 = head_inst->getRegOperand(matrix_inst, 1);
                matrix_interface->loadstoreMatrix(matrix_inst,o3cpu->getContext(head_inst->threadNumber), src1, src2);
                offload_minst = true;
                meminst_num--;
            } else if(matrix_inst->isMatrixConfig()){
                if(matrix_inst->getbits_31()){ //r-type
                    src1 = head_inst->getRegOperand(matrix_inst, 0);
                    matrix_interface->configMatrix(matrix_inst, o3cpu->getContext(head_inst->threadNumber), src1);
                } else{ //i-type
                    imm = matrix_inst->getbits_18_24();
                    DPRINTF(InstructionBuffer, "mcfgi imm = %llu\n", imm);
                    matrix_interface->configMatrix(matrix_inst, o3cpu->getContext(head_inst->threadNumber), imm);
                }
                offload_minst = true;
            } else if(matrix_inst->isMatrixInstArith() && !matrix_inst->ismzero()){
                matrix_interface->sendCommand(matrix_inst, o3cpu->getContext(head_inst->threadNumber));
                offload_minst = true;
            } else if (matrix_inst->ismzero()){
                DPRINTF(InstructionBuffer, "Mzero instruction detected!\n");
                matrix_interface->mzeroCmd(matrix_inst);
                offload_minst = true;
            }
        }
    }
    if(offload_minst)
        pop_inst();

}

void InstructionBuffer::set_cpu_ptr(gem5::o3::CPU* _o3cpu)
{
    o3cpu = _o3cpu;
}

} //namespace gem5

