/*
 * @Author: superboy
 * @Date: 2025-09-30 23:55:00
 * @LastEditTime: 2025-10-24 20:03:05
 * @LastEditors: superboy
 * @Description: 
 * @FilePath: /SJTU-matrix-engine/src/cpu/matrix_engine/inst_buf/inst_buf.cc
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
    buffer_stall
    .name(name() + ".buffer_stall")
    .desc("Number of Instruction Buffer stall cycle");

}

bool InstructionBuffer::push_inst(gem5::o3::DynInstPtr head_inst, uint64_t src1, uint64_t src2)
{
    // assert(!isFull());
    if(isFull()){
        return false;
    }
    inst_buf.push(head_inst);
    rs1_buf.push(src1);
    rs2_buf.push(src2);
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
    rs1_buf.pop();
    rs2_buf.pop();
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
        if(((matrix_inst->isStore() || matrix_inst->ismstorespm() || matrix_inst->isspmdmaload() || matrix_inst->isspmdmastore() || matrix_inst->ismsync_spm() || matrix_inst->isMatrixConfig()) && matrix_interface->requestGrant_withoutReg(matrix_inst))
        || ((!matrix_inst->isStore() && !matrix_inst->ismstorespm() && !matrix_inst->isspmdmaload() && !matrix_inst->isspmdmastore() && !matrix_inst->ismsync_spm() && !matrix_inst->isMatrixConfig()) && matrix_interface->requestGrant(matrix_inst))){
            DPRINTF(InstructionBuffer, "IB offload a matrix instruction to matrix engine!\n");
            uint64_t src1, src2;
            uint64_t imm;
            if(matrix_inst->isLoad()||matrix_inst->isStore()){
                src1 = rs1_buf.front();
                src2 = rs2_buf.front();
                matrix_interface->loadstoreMatrix(matrix_inst,o3cpu->getContext(head_inst->threadNumber), src1, src2);
                offload_minst = true;
                meminst_num--;
            } else if(matrix_inst->isMatrixConfig()){
                if(matrix_inst->getbits_25()){ //r-type
                    src1 = rs1_buf.front();
                    matrix_interface->configMatrix(matrix_inst, o3cpu->getContext(head_inst->threadNumber), src1);
                } else{ //i-type
                    imm = matrix_inst->uimm10();
                    DPRINTF(InstructionBuffer, "mcfgi imm = %llu\n", imm);
                    matrix_interface->configMatrix(matrix_inst, o3cpu->getContext(head_inst->threadNumber), imm);
                }
                offload_minst = true;
            } else if (matrix_inst->ismzero()){
                DPRINTF(InstructionBuffer, "Mzero instruction detected!\n");
                matrix_interface->mzeroCmd(matrix_inst, o3cpu->getContext(head_inst->threadNumber));
                offload_minst = true;
            } else if(matrix_inst->isMatrixInstArith() || (matrix_inst->ismredcadd() || matrix_inst->ismredcmax() || matrix_inst->ismlut())){
                matrix_interface->sendCommand(matrix_inst, o3cpu->getContext(head_inst->threadNumber));
                offload_minst = true;
            }  else if(matrix_inst->isMatrixDataMove()){
                if(matrix_inst->ismmovb_m_x()||matrix_inst->ismmovw_m_x()){
                    src1 = rs1_buf.front();
                    src2 = rs2_buf.front();
                    DPRINTF(InstructionBuffer, "%s, rs1 = %llu, rs2 = %llu\n", matrix_inst->getName(), src1, src2);
                    matrix_interface->scalar2matrixDataMove(matrix_inst, o3cpu->getContext(head_inst->threadNumber), src1, src2);
                    offload_minst = true;
                } else if(matrix_inst->ismmovb_x_m()||matrix_inst->ismmovw_x_m()){
                    src1 = rs1_buf.front();
                    src2 = matrix_inst->mrd();
                    DPRINTF(InstructionBuffer, "%s, rs1 = %llu\n", matrix_inst->getName(), src1);
                    matrix_interface->matrix2scalarDataMove(matrix_inst, o3cpu->getContext(head_inst->threadNumber), src1,
                        [this,head_inst,matrix_inst,src2](uint64_t val) mutable {
                            DPRINTF(InstructionBuffer,"The instruction %s has been hosted by the Matrix Engine\n", matrix_inst->getName());
                            head_inst->setRegOperand(matrix_inst, 0, val);
                    });
                    offload_minst = true;
                } else if(matrix_inst->ismmov_mm()){
                    matrix_interface->matrixDataMove(matrix_inst, o3cpu->getContext(head_inst->threadNumber));
                    offload_minst = true;
                } else {
                    DPRINTF(InstructionBuffer, "Matrix data move type error!\n");
                    offload_minst = true; 
                }
            } else if(matrix_inst->isMatrixBroadcast()){
                imm = matrix_inst->getbits_23_25();
                DPRINTF(InstructionBuffer, "%s, imm = %llu\n", matrix_inst->getName(), imm);
                matrix_interface->broadcastMatrix(matrix_inst, o3cpu->getContext(head_inst->threadNumber), imm);
                offload_minst = true;
            } else if(matrix_inst->ismloadspm()||matrix_inst->ismstorespm()||matrix_inst->isspmdmaload()||matrix_inst->isspmdmastore()){
                DPRINTF(InstructionBuffer, "SPM instructions detected!! \n");
                src1 = rs1_buf.front();
                src2 = rs2_buf.front();
                DPRINTF(InstructionBuffer, "SPM instructions src1 = %d, src2 = %d\n", src1, src2);
                DPRINTF(InstructionBuffer, "SPM instructions src1 = %d, src2 = %d\n", matrix_inst->mrs1(), matrix_inst->mrs2());
                matrix_interface->sendSpmCommand(matrix_inst, src1, src2, o3cpu->getContext(head_inst->threadNumber));
                offload_minst = true;
                meminst_num--;
            } else if(matrix_inst->ismsync_spm()){
                DPRINTF(InstructionBuffer, "SPM SYNC instruction detected!! \n");
                matrix_interface->sendSpmSync(matrix_inst, o3cpu->getContext(head_inst->threadNumber));
                offload_minst = true;
            } else {
                matrix_interface->sendCommand(matrix_inst, o3cpu->getContext(head_inst->threadNumber));
                offload_minst = true; 
            }
        }
    }
    if(offload_minst)
        pop_inst();
    else
        buffer_stall++;

}

void InstructionBuffer::set_cpu_ptr(gem5::o3::CPU* _o3cpu)
{
    o3cpu = _o3cpu;
}

} //namespace gem5

