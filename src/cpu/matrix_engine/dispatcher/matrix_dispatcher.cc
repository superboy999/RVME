// ==========================================
// Author: cwq
// Last Date: 2024/2/28
// Description: Decode, and Will send to
// queue first, reorder and rename! Finally 
// send to functional unit!
// ==========================================

#include "cpu/matrix_engine/dispatcher/matrix_dispatcher.hh"
#include "cpu/matrix_engine/spm/matrix_dma.hh"
#include "debug/MatrixDispatcher.hh"
#include "debug/MatrixDispatcherReadLock.hh"

#include <cassert>
#include <cstdint>
#include <string>

namespace gem5
{
class DmaReqState;

MatrixDispatcher::MatrixDispatcher(const MatrixDispatcherParams &params) :
TickedObject(params), tileReg_num(params.tileReg_num), accReg_num(params.accReg_num), MQ_depth(params.MQ_depth), AQ_depth(params.AQ_depth), 
OoO(true)
{
    // config_stall = false;
    // dispatchReq = false;
    // issueReq = false;
    // dispatchArithReq = false;
    // dispatchMemReq = false;
    issueArithReq = false;
    issueMemReq = false;
    XY = false;
    loada = 0, loadb = 0, loadc = 0, loadacommit = 0, loadbcommit = 0, loadccommit = 0;
    transfer_num = 0;
    // sizeM = 0;
    // sizeN = 0;
    // sizeK = 0;
    isBusy = false;
}
MatrixDispatcher::~MatrixDispatcher(){}

void MatrixDispatcher::set_matrixEnginePtr(MatrixEngine* _matrix_engine)
{
    matrix_engine = _matrix_engine;
}

void MatrixDispatcher::regStats()
{
    TickedObject::regStats();
    WaitingForReg
    .name(name() + ".TicksForWaitingReg")
    .desc("Ticks of waiting for free physical register");
    WaitingForLane
    .name(name() + ".TicksForWaitingLane")
    .desc("Ticks of waiting for free Matrix Lane");
    WaitingForEWU
    .name(name() + ".TicksForWaitingEWU")
    .desc("Ticks of waiting for free EWU");
    allLaneisFree
    .name(name() + ".TicksForAllLaneFree")
    .desc("Ticks of all Matrix Lane is free");
    LaneisFree
    .name(name() + ".TicksForLaneFree")
    .desc("Ticks of Matrix Lane is free");
    LaneWaitingForReg
    .name(name() + ".TicksForLaneWaitingReg")
    .desc("Ticks of Lane waiting for reg");
    hasFreeTileReg
    .name(name() + ".TicksForFreeReg")
    .desc("Ticks of free Tile reg");
    hasFreeAccReg
    .name(name() + ".TicksForFreeAccReg")
    .desc("Ticks of free Acc reg");
    MatrixMQentryUsed
    .name(name() + ".MatrixMQentryUsed")
    .desc("Number of Matrix Memory Queue entry used!");
    MatrixAQentryUsed
    .name(name() + ".MatrixAQentryUsed")
    .desc("Number of Matrix Arith Queue entry used!");
    AQread
    .name(name() + ".AQread")
    .desc("Number of Arith Queue read");
    AQwrite
    .name(name() + ".AQwrite")
    .desc("Number of Arith Queue write");
    MQread
    .name(name() + ".MQread")
    .desc("Number of Memory Queue read");
    MQwrite
    .name(name() + ".MQwrite")
    .desc("Number of Memory Queue write");
    matrix_load_inst
    .name(name() + ".matrix_load_inst")
    .desc("Number of matrix load instruction");
    matrix_store_inst
    .name(name() + ".matrix_store_inst")
    .desc("Number of matrix store instruction");
    IssueNum
    .name(name() + ".IssueNum")
    .desc("Number of instructions");
    testnum
    .name(name() + ".testnum")
    .desc("testnum");   
}

void MatrixDispatcher::renameMatrixInst(RiscvISA::RiscvMatrixInst &minst, ScoreBoard_Entry* matrix_sbe, ThreadContext* tc)
{
    uint16_t pms1;
    uint16_t pms2;
    uint16_t pms3;
    uint16_t pmd;
    uint16_t old_dst;

    uint16_t ms1 = minst.ms1();
    uint16_t ms2 = minst.ms2();
    uint16_t ms3 = minst.ms3();
    uint16_t md = minst.md();
    uint16_t load_md = minst.md();
    matrix_sbe->set_MatrixStaticInst(&minst);
    matrix_sbe->ms1 = ms1;
    matrix_sbe->ms2 = ms2;
    matrix_sbe->ms3 = ms3;
    matrix_sbe->md = md;
    // matrix_engine->matrix_rename->set_RAT_vld(load_md, false); //FIXED: every md before inst finished, will be marked as invalid right behind selected!

    if(minst.isLoad()||minst.isStore()||minst.ismloadspm()||minst.ismstorespm()||minst.isspmdmaload()||minst.isspmdmastore()||minst.ismmov_mm()){
        if(minst.isLoad()){
            //This function has destreg, so rename here.
            // mld<b/h/w/d> md, rs2, (rs1)
            if(load_md < 4){
                pmd = matrix_engine->matrix_rename->get_freeTileReg();
            } else if (load_md < 8 && load_md >= 4){
                pmd = matrix_engine->matrix_rename->get_freeAccReg();
            }
            // pmd = matrix_engine->matrix_rename->get_freeReg();
            matrix_engine->matrix_rename->regLock(pmd);
            old_dst = matrix_engine->matrix_rename->get_preg_RAT(load_md);
            matrix_engine->matrix_rename->set_preg_RAT(load_md, pmd);
            matrix_sbe->set_dst_lrf_num(load_md);
            matrix_sbe->set_dst_prf_num(pmd);
            matrix_sbe->set_old_dst(old_dst);
            matrix_load_inst++;
            matrix_sbe->dst_memdep_idx = matrix_engine->matrix_rob->set_md_entry(load_md);
            DPRINTF(MatrixDispatcher, "load set md entry, idx = %d, dst_memdep_idx = %d\n", load_md, matrix_sbe->dst_memdep_idx);
        } else if(minst.isStore()){
            //This function has no destination register, so dont rename;
            //mst<b/h/w/d> ms3, rs2, (rs1)
            pms3 = matrix_engine->matrix_rename->get_preg_RAT(ms3);
            // matrix_engine->matrix_rename->regLock(pms3);
            matrix_sbe->set_renamed_src3(pms3);
            matrix_store_inst++;
            matrix_sbe->src3_memdep_idx = matrix_engine->matrix_rob->mdtail[ms3];
            matrix_sbe->src3_read_idx = matrix_engine->matrix_rename->readLock(pms3);
        } else if(minst.ismloadspm()){
            // DPRINTF(MatrixDispatcher, "SPM LOAD instruction detected in rename stage!\n");
            matrix_sbe->set_spm_addr(matrix_sbe->get_rs1_value());
            matrix_sbe->set_stride(matrix_sbe->get_rs2_value());
            matrix_sbe->set_setid(minst.bit29_28());
            matrix_sbe->set_mode(minst.spm_mode());
            matrix_sbe->set_rows(minst.bit29_28() == 0 ? matrix_engine->get_sizeM() : minst.bit29_28() == 1 ? matrix_engine->get_sizeK() : matrix_engine->get_sizeM());
            matrix_sbe->set_row_size(minst.bit29_28() == 0 ? 32 : minst.bit29_28() == 1 ? 8 : 32);
            // DPRINTF(MatrixDispatcher, "Set spm_addr = %u, stride = %u, setid = %u, mode = %u, rows = %u, row_size = %u\n", matrix_sbe->get_spm_addr(), matrix_sbe->get_stride(),
            //     matrix_sbe->get_setid(), matrix_sbe->get_mode(), matrix_sbe->get_rows(), matrix_sbe->get_row_size());
            ///////////// same as load //////////////////
            if(load_md < 4){
                pmd = matrix_engine->matrix_rename->get_freeTileReg();
            } else if (load_md < 8 && load_md >= 4){
                pmd = matrix_engine->matrix_rename->get_freeAccReg();
            }
            if(matrix_sbe->get_setid() == 0) loada++;
            else if(matrix_sbe->get_setid() == 1) loadb++;
            else if(matrix_sbe->get_setid() == 2) loadc++;
            // DPRINTF(MatrixDispatcher, "mload a = %d, load b = %d, load c = %d\n", loada, loadb, loadc);
            // pmd = matrix_engine->matrix_rename->get_freeReg();
            matrix_engine->matrix_rename->regLock(pmd);
            old_dst = matrix_engine->matrix_rename->get_preg_RAT(load_md);
            matrix_engine->matrix_rename->set_preg_RAT(load_md, pmd);
            matrix_sbe->set_dst_lrf_num(load_md);
            matrix_sbe->set_dst_prf_num(pmd);
            matrix_sbe->set_old_dst(old_dst);
            matrix_load_inst++;
            matrix_sbe->dst_memdep_idx = matrix_engine->matrix_rob->set_md_entry(load_md);
            DPRINTF(MatrixDispatcher, "load spm set md entry, pmd = %d, idx = %d, dst_memdep_idx = %d\n", pmd, load_md, matrix_sbe->dst_memdep_idx);
            auto spmReq = std::make_shared<DmaReqState>(false, false, *tc);
            matrix_sbe->setDmaReqState(spmReq);
            matrix_sbe->spm_reqId = matrix_engine->matrix_spm->sendMELoadSPM(spmReq, minst.bit29_28());
            ///////////// same as load //////////////////
        } else if(minst.ismstorespm()) {
            // DPRINTF(MatrixDispatcher, "SPM STORE instruction detected in rename stage!\n");
            matrix_sbe->set_spm_addr(matrix_sbe->get_rs1_value());
            matrix_sbe->set_stride(matrix_sbe->get_rs2_value());
            matrix_sbe->set_setid(minst.bit29_28());
            matrix_sbe->set_mode(minst.spm_mode());
            matrix_sbe->set_rows(minst.bit29_28() == 0 ? matrix_engine->get_sizeM() : minst.bit29_28() == 1 ? matrix_engine->get_sizeK() : matrix_engine->get_sizeM());
            // matrix_sbe->set_row_size(minst.bit29_28() == 0 ? matrix_engine->get_sizeK() : minst.bit29_28() == 1 ? matrix_engine->get_sizeN() : 4 * matrix_engine->get_sizeN());
            matrix_sbe->set_row_size(minst.bit29_28() == 0 ? 32 : minst.bit29_28() == 1 ? 8 : 32);
            // DPRINTF(MatrixDispatcher, "Set spm_addr = %u, stride = %u, setid = %u, mode = %u, rows = %u, row_size = %u\n", matrix_sbe->get_spm_addr(), matrix_sbe->get_stride(),
            //     matrix_sbe->get_setid(), matrix_sbe->get_mode(), matrix_sbe->get_rows(), matrix_sbe->get_row_size());
            ///////////// same as store //////////////////
            pms3 = matrix_engine->matrix_rename->get_preg_RAT(ms3);
            // matrix_engine->matrix_rename->regLock(pms3);
            matrix_sbe->set_renamed_src3(pms3);
            matrix_store_inst++;
            matrix_sbe->src3_memdep_idx = matrix_engine->matrix_rob->mdtail[ms3];
            auto spmReq = std::make_shared<DmaReqState>(false, false, *tc);
            matrix_sbe->setDmaReqState(spmReq);
            matrix_sbe->spm_reqId = matrix_engine->matrix_spm->sendMEStoreSPM(spmReq, minst.bit29_28());
            matrix_sbe->src3_read_idx = matrix_engine->matrix_rename->readLock(pms3);
            DPRINTF(MatrixDispatcherReadLock, "StoreSPM read lock, idx = %d, dst_memdep_idx = %d, pms3 = %d\n", load_md, matrix_sbe->dst_memdep_idx, pms3);
            ///////////// same as store //////////////////
        } else if(minst.isspmdmaload()) {
            // DPRINTF(MatrixDispatcher, "SPM DMA LOAD instruction detected in rename stage!\n");
            matrix_sbe->set_mem_addr(matrix_sbe->get_rs1_value());
            matrix_sbe->set_spm_addr(matrix_sbe->get_rs2_value());
            // DPRINTF(MatrixDispatcher, "SPM sendTimingReadReq, rs1 = %d, rs2 = %d\n", matrix_sbe->get_rs1_value(), matrix_sbe->get_rs2_value());
            matrix_sbe->set_spm_uimm5(minst.uimm5());
            auto spmReq = std::make_shared<DmaReqState>(MemCmd::ReadReq, matrix_sbe->get_spm_addr(), matrix_sbe->get_mem_addr(), matrix_sbe->get_spm_uimm5(), *tc);
            matrix_sbe->setDmaReqState(spmReq);
            matrix_engine->matrix_spm->sendTimingReadReq(matrix_sbe->get_mem_addr(), matrix_sbe->get_spm_addr(), matrix_sbe->get_spm_uimm5(), spmReq);
            // DPRINTF(MatrixDispatcher, "SPM sendTimingReadReq, matrix_sbe->get_mem_addr() = %d, matrix_sbe->get_spm_addr() = %d, matrix_sbe->get_spm_uimm5() = %d, spmReq = %d\n",
            //     matrix_sbe->get_mem_addr(), matrix_sbe->get_spm_addr(), matrix_sbe->get_spm_uimm5(), spmReq);
            matrix_engine->matrix_spm->meminst_num++;
        } else if(minst.isspmdmastore()) {
            // DPRINTF(MatrixDispatcher, "SPM DMA STORE instruction detected in rename stage!\n");
            matrix_sbe->set_mem_addr(matrix_sbe->get_rs1_value());
            matrix_sbe->set_spm_addr(matrix_sbe->get_rs2_value());
            matrix_sbe->set_spm_uimm5(minst.uimm5());
            auto spmReq = std::make_shared<DmaReqState>(MemCmd::WriteReq, matrix_sbe->get_spm_addr(), matrix_sbe->get_mem_addr(), matrix_sbe->get_spm_uimm5(), *tc);
            matrix_sbe->setDmaReqState(spmReq);
            matrix_engine->matrix_spm->sendTimingWriteReq(matrix_sbe->get_mem_addr(), matrix_sbe->get_spm_addr(), matrix_sbe->get_spm_uimm5(), spmReq);
            matrix_engine->matrix_spm->meminst_num++;
        } else if(minst.ismmov_mm()) {
            if(load_md < 4){
                pmd = matrix_engine->matrix_rename->get_freeTileReg();
            } else if (load_md < 8 && load_md >= 4){
                pmd = matrix_engine->matrix_rename->get_freeAccReg();
            }
            // pmd = matrix_engine->matrix_rename->get_freeReg();
            matrix_engine->matrix_rename->regLock(pmd);
            old_dst = matrix_engine->matrix_rename->get_preg_RAT(load_md);
            matrix_engine->matrix_rename->set_preg_RAT(load_md, pmd);
            matrix_sbe->set_dst_lrf_num(load_md);
            matrix_sbe->set_dst_prf_num(pmd);
            matrix_sbe->set_old_dst(old_dst);
            matrix_sbe->dst_memdep_idx = matrix_engine->matrix_rob->set_md_entry(load_md);
            pms1 = matrix_engine->matrix_rename->get_preg_RAT(ms1);
            matrix_engine->matrix_rename->regLock(pms1);
            matrix_sbe->set_renamed_src1(pms1);
            matrix_sbe->src1_memdep_idx = matrix_engine->matrix_rob->mdtail[ms1];
            matrix_sbe->src1_read_idx = matrix_engine->matrix_rename->readLock(pms1);
        }
        matrix_sbe->set_cfg_size(matrix_engine->get_sizeN(), matrix_engine->get_sizeM(), matrix_engine->get_sizeK());
    } else if(minst.ismmovb_m_x() || minst.ismmovb_x_m() || minst.ismmovw_m_x() || minst.ismmovw_x_m()) {
        if(minst.ismmovb_m_x() || minst.ismmovw_m_x()) {
            if(load_md < 4){
                pmd = matrix_engine->matrix_rename->get_freeTileReg();
            } else if (load_md < 8 && load_md >= 4){
                pmd = matrix_engine->matrix_rename->get_freeAccReg();
            }
            matrix_engine->matrix_rename->regLock(pmd);
            old_dst = matrix_engine->matrix_rename->get_preg_RAT(load_md);
            matrix_engine->matrix_rename->set_preg_RAT(load_md, pmd);
            matrix_sbe->set_dst_lrf_num(load_md);
            matrix_sbe->set_dst_prf_num(pmd);
            matrix_sbe->set_old_dst(old_dst);
            matrix_sbe->dst_memdep_idx = matrix_engine->matrix_rob->set_md_entry(load_md);
        }
        else {
            pms2 = matrix_engine->matrix_rename->get_preg_RAT(ms2);
            matrix_engine->matrix_rename->regLock(pms2);
            matrix_sbe->set_renamed_src2(pms2);
            matrix_sbe->src2_memdep_idx = matrix_engine->matrix_rob->mdtail[ms2];
            matrix_sbe->src2_read_idx = matrix_engine->matrix_rename->readLock(pms2);
        }
        matrix_sbe->set_cfg_size(matrix_engine->get_sizeN(), matrix_engine->get_sizeM(), matrix_engine->get_sizeK());
    } else if(minst.ismzero()) {
        if(load_md < 4){
            pmd = matrix_engine->matrix_rename->get_freeTileReg();
        } else if (load_md < 8 && load_md >= 4){
            pmd = matrix_engine->matrix_rename->get_freeAccReg();
        }
        matrix_engine->matrix_rename->regLock(pmd);
        old_dst = matrix_engine->matrix_rename->get_preg_RAT(load_md);
        matrix_engine->matrix_rename->set_preg_RAT(load_md, pmd);
        matrix_sbe->set_dst_lrf_num(load_md);
        matrix_sbe->set_dst_prf_num(pmd);
        matrix_sbe->set_old_dst(old_dst);
        matrix_sbe->dst_memdep_idx = matrix_engine->matrix_rob->set_md_entry(load_md);
        matrix_sbe->set_cfg_size(matrix_engine->get_sizeN(), matrix_engine->get_sizeM(), matrix_engine->get_sizeK());
    }
    else if(minst.isMatrixInstArith()){
        // This function has destination
        // #8bit data width
        // #signed matrix multiply
        // mmaqa.b md, ms2, ms1
        // #unsigned matrix multiply
        // mmaqau.b md, ms2, ms1
        // #unsigned-signed matrix multiply
        // mmaqaus.b md, ms2, ms1
        // #signed-unsigned matrix multiply
        // mmaqasu.b md, ms2, ms1

        // Note: Here is special: For accumulator register, only mzero can rename it, or is will coninuelly save to same physical register!
        if(minst.ismmacc()){
            if(md < 4){
                // pmd = matrix_engine->matrix_rename->get_freeTileReg();
                panic("THIS is A FAULT INSTRUCTION");
            } else if (md < 8 && md >= 4){
                if(matrix_engine->matrix_rename->accCanRls(md)){
                    matrix_engine->matrix_rename->setAcc(md, false); //mark it as used
                    pmd = matrix_engine->matrix_rename->get_freeAccReg();
                    matrix_engine->matrix_rename->regLock(pmd);
                    old_dst = matrix_engine->matrix_rename->get_preg_RAT(md);
                    matrix_engine->matrix_rename->set_preg_RAT(md, pmd);
                    matrix_engine->matrix_rename->set_PR_vld(pmd, true); //mark the new acc register as valid, 为了让一开始acc操作能读取自己这个acc寄存器跟自己相加!!!注意要放在setpregrat后面，保证置为valid
                } else {
                    pmd = matrix_engine->matrix_rename->get_preg_RAT(md);
                    matrix_engine->matrix_rename->regLock(pmd);
                    old_dst = 99; //没有更换目标寄存器
                }
                // pmd = matrix_engine->matrix_rename->get_freeAccReg();
                matrix_sbe->dst_memdep_idx = matrix_engine->matrix_rob->set_md_entry(md);
                DPRINTF(MatrixDispatcher, "mac set md entry, idx = %d, pmd = %d, dst_memdep_idx = %d\n", md, pmd, matrix_sbe->dst_memdep_idx);
                matrix_sbe->src1_memdep_idx = matrix_engine->matrix_rob->mdtail[ms1];
                matrix_sbe->src2_memdep_idx = matrix_engine->matrix_rob->mdtail[ms2];
                matrix_sbe->src3_memdep_idx = matrix_engine->matrix_rob->mdtail[md]; //special! src3 represent C in D = A*B + C
                matrix_sbe->src3_read_idx = matrix_engine->matrix_rename->get_readLock(pmd);
            }

            pms1 = matrix_engine->matrix_rename->get_preg_RAT(ms1);
            pms2 = matrix_engine->matrix_rename->get_preg_RAT(ms2);
            matrix_engine->matrix_rename->regLock(pms1);
            matrix_engine->matrix_rename->regLock(pms2);
            matrix_sbe->set_renamed_src1(pms1);
            matrix_sbe->set_renamed_src2(pms2);
            matrix_sbe->set_dst_lrf_num(md);
            matrix_sbe->set_dst_prf_num(pmd);
            matrix_sbe->set_old_dst(old_dst);

            matrix_sbe->set_cfg_size(matrix_engine->get_sizeN(), matrix_engine->get_sizeM(), matrix_engine->get_sizeK());
        } else if (minst.ismadd()||minst.ismsub()||minst.ismmul()||minst.ismmulh()||minst.ismmax()||minst.ismumax()||minst.ismmin()||minst.ismumin()||minst.ismsll()||minst.ismsrl()||minst.ismsra()){
            if(md < 4)
                panic("THIS is A FAULT INSTRUCTION");
            else if(md < 8 && md >= 4){
                pms1 = matrix_engine->matrix_rename->get_preg_RAT(ms1);
                pms2 = matrix_engine->matrix_rename->get_preg_RAT(ms2);
                matrix_engine->matrix_rename->regLock(pms1);
                matrix_engine->matrix_rename->regLock(pms2);
                matrix_sbe->set_renamed_src1(pms1);
                matrix_sbe->set_renamed_src2(pms2);
                matrix_sbe->src1_memdep_idx = matrix_engine->matrix_rob->mdtail[ms1];
                matrix_sbe->src2_memdep_idx = matrix_engine->matrix_rob->mdtail[ms2];
                if(ms1 == md || ms2 == md) {
                    pmd = matrix_engine->matrix_rename->get_preg_RAT(md);
                    matrix_engine->matrix_rename->regLock(pmd);
                    old_dst = 99;
                }                    
                else {
                    pmd = matrix_engine->matrix_rename->get_freeAccReg();
                    matrix_engine->matrix_rename->regLock(pmd);
                    old_dst = matrix_engine->matrix_rename->get_preg_RAT(md);
                    matrix_engine->matrix_rename->set_preg_RAT(md, pmd);
                }                    
                matrix_sbe->set_dst_lrf_num(md);
                matrix_sbe->set_dst_prf_num(pmd);
                matrix_sbe->set_old_dst(old_dst);

                matrix_sbe->set_cfg_size(matrix_engine->get_sizeN(), matrix_engine->get_sizeM(), matrix_engine->get_sizeK());
            }
            matrix_sbe->dst_memdep_idx = matrix_engine->matrix_rob->set_md_entry(md);
            if(pms1 != pmd) {
                matrix_sbe->src1_read_idx = matrix_engine->matrix_rename->readLock(pms1);
                DPRINTF(MatrixDispatcherReadLock, "Elementwise read lock, idx = %d, dst_memdep_idx = %d, read_idx = %d\n", load_md, matrix_sbe->dst_memdep_idx, matrix_sbe->src1_read_idx);
            }
            if(pms2 != pmd) {
                matrix_sbe->src2_read_idx = matrix_engine->matrix_rename->readLock(pms2);
                DPRINTF(MatrixDispatcherReadLock, "Elementwise read lock, idx = %d, dst_memdep_idx = %d, read_idx = %d\n", load_md, matrix_sbe->dst_memdep_idx, matrix_sbe->src2_read_idx);
            }
            if(pms1 == pmd || pms2 == pmd) {
                matrix_sbe->dst_read_idx = matrix_engine->matrix_rename->get_readLock(pmd);
            }
            DPRINTF(MatrixDispatcher, "unit set md entry, pmd = %d, idx = %d, dst_memdep_idx = %d\n", pmd, md, matrix_sbe->dst_memdep_idx);
        }
    } else if (minst.ismredcadd() || minst.ismredcmax() || minst.ismlut()) {
            if(md < 4)
                panic("THIS is A FAULT INSTRUCTION");
            else if(md < 8 && md >= 4){
                pms1 = matrix_engine->matrix_rename->get_preg_RAT(ms1);
                matrix_engine->matrix_rename->regLock(pms1);
                matrix_sbe->set_renamed_src1(pms1);
                matrix_sbe->src1_memdep_idx = matrix_engine->matrix_rob->mdtail[ms1];
                if(ms1 == md) {
                    pmd = matrix_engine->matrix_rename->get_preg_RAT(md);
                    matrix_engine->matrix_rename->regLock(pmd);
                    old_dst = 99;
                }
                else {
                    pmd = matrix_engine->matrix_rename->get_freeAccReg();
                    matrix_engine->matrix_rename->regLock(pmd);
                    old_dst = matrix_engine->matrix_rename->get_preg_RAT(md);
                    matrix_engine->matrix_rename->set_preg_RAT(md, pmd);
                }
                matrix_sbe->set_dst_lrf_num(md);
                matrix_sbe->set_dst_prf_num(pmd);
                matrix_sbe->set_old_dst(old_dst);

                matrix_sbe->set_cfg_size(matrix_engine->get_sizeN(), matrix_engine->get_sizeM(), matrix_engine->get_sizeK());
            }
            matrix_sbe->dst_memdep_idx = matrix_engine->matrix_rob->set_md_entry(md);
            if(pms1 != pmd) {
                matrix_sbe->src1_read_idx = matrix_engine->matrix_rename->readLock(pms1);
                DPRINTF(MatrixDispatcherReadLock, "Redc read lock, idx = %d, dst_memdep_idx = %d, read_idx = %d\n", load_md, matrix_sbe->dst_memdep_idx, matrix_sbe->src1_read_idx);
            }
            else
                matrix_sbe->dst_read_idx = matrix_engine->matrix_rename->get_readLock(pmd);
            DPRINTF(MatrixDispatcher, "unit set md entry, idx = %d, dst_memdep_idx = %d\n", md, matrix_sbe->dst_memdep_idx);
        } 
        //  else if (minst.ismzero()){
        //     matrix_engine->matrix_rename->setAcc(md, true);
        // }
}

void MatrixDispatcher::set_rob_entry(RiscvISA::RiscvMatrixInst &minst, ScoreBoard_Entry* matrix_sbe)
{
    assert(!matrix_engine->matrix_rob->rob_full());
    if(!matrix_engine->matrix_rob->isOccupied()){
        matrix_engine->matrix_rob->startTicking();
    }
    bool old_dst_vld = false;
    if(minst.isStore() || minst.ismstorespm() || minst.ismmovb_x_m() || minst.ismmovw_x_m()){
        old_dst_vld = false;
    } else{
        old_dst_vld = true;
    }
    if(!minst.isStore() && !minst.ismstorespm() && !minst.ismmovb_x_m() && !minst.ismmovw_x_m()){
        uint32_t rob_entry = matrix_engine->matrix_rob->set_rob_entry(matrix_sbe->get_dst_lrf_num(), matrix_sbe->get_dst_prf_num(), matrix_sbe->get_old_dst(), old_dst_vld);
        matrix_sbe->set_rob_entry(rob_entry);
    }

    if(minst.isLoad() || minst.isStore() || minst.ismloadspm()||minst.ismstorespm()){
        matrix_engine->matrix_rob->meminst_num++;
    }
}

void MatrixDispatcher::sendInstToCQ(ScoreBoard_Entry* matrix_sbe, uint64_t src1_value, ThreadContext *tc)
{
    //FIXME: add code to the matrix_engine configMatrix function;
    // matrix_engine->configMatrix(minst, src1_value, *tc);
    // assert(cfg_Queue.size()==0);
    cfg_Queue.push_back(matrix_sbe);
    if(cfg_Queue.size()!=0){
        ScoreBoard_Entry* matrix_sbe = cfg_Queue.back();
        if(matrix_sbe->_minst->ismcfgki()){
            matrix_engine->cfgSizeK(matrix_sbe->_minst->uimm10()); 
        }
        if(matrix_sbe->_minst->ismcfgmi()){
            matrix_engine->cfgSizeM(matrix_sbe->_minst->uimm10());
        }
        if(matrix_sbe->_minst->ismcfgni()){
            matrix_engine->cfgSizeN(matrix_sbe->_minst->uimm10());
        }
        if(matrix_sbe->_minst->ismcfgk()){
            matrix_engine->cfgSizeK(matrix_sbe->get_rs1_value());
        }
        if(matrix_sbe->_minst->ismcfgm()){
            matrix_engine->cfgSizeM(matrix_sbe->get_rs1_value());
        }
        if(matrix_sbe->_minst->ismcfgn()){
            matrix_engine->cfgSizeN(matrix_sbe->get_rs1_value());
        }
        if(matrix_sbe->_minst->ismcfg()){
            matrix_engine->cfgSizeN(matrix_sbe->_minst->getbits_8_15());
            matrix_engine->cfgSizeM(matrix_sbe->_minst->getbits_0_7());
            matrix_engine->cfgSizeK(matrix_sbe->_minst->getbits_16_31());
        }

        cfg_Queue.pop_back();
            // delete matrix_sbe;
            // config_stall = false;
        }

    // config_stall = true;
}

void MatrixDispatcher::recvMzero(ScoreBoard_Entry* matrix_sbe)
{
    DPRINTF(MatrixDispatcher, "recv mzero inst md = %d\n", matrix_sbe->_minst->md());
}

void MatrixDispatcher::sendInstToMQ(ScoreBoard_Entry* matrix_sbe, uint64_t src1_value, uint64_t src2_value, ThreadContext *tc)
{
    MemQueueEntry* mqe = new MemQueueEntry(*matrix_sbe, src1_value, src2_value, *tc);
    Memory_Queue.push_back(mqe);
    MQwrite++;
}

void MatrixDispatcher::sendInstToAQ(ScoreBoard_Entry* matrix_sbe, ThreadContext *tc)
{
    uint16_t ms1 = matrix_sbe->_minst->ms1();
    uint16_t ms2 = matrix_sbe->_minst->ms2();
    uint16_t md = matrix_sbe->_minst->md();
    ArithQueueEntry* aqe = new ArithQueueEntry(*matrix_sbe, ms1, ms2, md, *tc);
    Arithmetic_Queue.push_back(aqe);
    // printf("send to aq, size = %d, ms1 = %d, ms2 = %d", Arithmetic_Queue.size(), Arithmetic_Queue[0]->ms1, Arithmetic_Queue[0]->ms2);
}

// bool MatrixDispatcher::dispatchGrant(RiscvISA::RiscvMatrixInst &minst)
// {
//     //=======================
//     //Send to inst/mem queue||
//     //=======================
//     if(minst.isLoad()||minst.isStore())
//     {
//         if(Memory_Queue.size()<=MQ_depth){
//             // dispatchMemReq = true;
//             // minst_temp = minst;
//             return true;
//         } else{
//             return false;
//         }
//     } else if(minst.isMatrixInstArith()){
//         if(Arithmetic_Queue.size()<=AQ_depth){
//             // minst_temp = minst;
//             // dispatchArithReq = true;
//             return true;
//         } else{
//             return false;
//         }
//     } else if (minst.isMatrixConfig()){
//         config_stall = true;
//         return true; //FIXME: Maybe there is some stall here?
//     } else if (config_stall){
//         return false;
//     }
//     // assert (!dispatchReq);
//     // dispatchReq = true;
// }

bool MatrixDispatcher::dispatchGrant(RiscvISA::RiscvMatrixInst &minst)
{
    bool rob_entry_notfull = !matrix_engine->matrix_rob->rob_full();
    bool queue_notfull = false;
    if(minst.isLoad()||minst.isStore()||minst.ismloadspm()||minst.ismstorespm()||minst.ismmov_mm()||minst.ismmovb_m_x()||minst.ismmovb_x_m()||minst.ismmovw_m_x()||minst.ismmovw_x_m()||minst.ismzero())
    {
        if(Memory_Queue.size()<=MQ_depth){
            queue_notfull = true;
        }
    } else if(minst.isMatrixInstArith() || minst.isMatrixRedc() || minst.ismlut()){
        if(Arithmetic_Queue.size()<=AQ_depth){
            // DPRINTF(MatrixDispatcher, "AQ size = %d\n", Arithmetic_Queue.size());
            // DPRINTF(MatrixDispatcher, "AQ_depth = \n", AQ_depth);
            assert(Arithmetic_Queue.size()<=AQ_depth);
            queue_notfull = true;
        }
    } else if (minst.isMatrixConfig()){
        // config_stall = true;
        queue_notfull = true; //FIXME: Maybe there is some stall here?
        XY = false;
    }
    // bool free_prf_avaliable = !matrix_engine->matrix_rename->freeList_empty();
    bool free_prf_avaliable = false;
    if(minst.md() < 4){
        free_prf_avaliable = !matrix_engine->matrix_rename->freeTileList_empty();
    } else if (minst.md() < 8 && minst.md() >=4){
        free_prf_avaliable = !matrix_engine->matrix_rename->freeAccList_empty();
    }
    // DPRINTF(MatrixDispatcher, "rob:%d, M/AQ:%d, free_pr:%d\n", rob_entry_notfull, queue_notfull, free_prf_avaliable);
    if(!free_prf_avaliable){
        WaitingForReg++;
    }
    return rob_entry_notfull&&queue_notfull&&free_prf_avaliable; //Delete config stall
    // return rob_entry_notfull&&queue_notfull&&free_prf_avaliable&&!config_stall;//FIXME: check config_stall correctness.
}

bool MatrixDispatcher::dispatchGrant_withoutReg(RiscvISA::RiscvMatrixInst &minst)
{
    if(minst.ismsync_spm()) {
        return true;
    }
    if (minst.isspmdmaload() || minst.isspmdmastore()) {
        return true;
    }
    bool rob_entry_notfull = !matrix_engine->matrix_rob->rob_full();
    bool queue_notfull = false;
    if(minst.isLoad()||minst.isStore()||minst.ismloadspm()||minst.ismstorespm()||minst.ismmov_mm()||minst.ismmovb_m_x()||minst.ismmovb_x_m()||minst.ismmovw_m_x()||minst.ismmovw_x_m())
    {
        if(Memory_Queue.size()<=MQ_depth){
            queue_notfull = true;
        }
    } else if(minst.isMatrixInstArith()){
        if(Arithmetic_Queue.size()<=AQ_depth){
            queue_notfull = true;
        }
    } else if (minst.isMatrixConfig()){
        // config_stall = true;
        queue_notfull = true; //FIXME: Maybe there is some stall here?
        XY = false;
    }
    // DPRINTF(MatrixDispatcher, "rob:%d, M/AQ:%d\n", rob_entry_notfull, queue_notfull);

    return rob_entry_notfull&&queue_notfull; //Delete config stall
    // return rob_entry_notfull&&queue_notfull&&free_prf_avaliable&&!config_stall;//FIXME: check config_stall correctness.
}

// bool MatrixDispatcher::issueGrant()
// {
//     //=========================================================
//     //Send inst from mem/inst queue to specific function unit||
//     //=========================================================

//     这块要加上rename还有空寄存器以及reoreder还有空间
    
//     bool grant = false;
//     if(Memory_Queue.size() > 0)
//     {
//         issueMemReq = true;
//         return true;
//     } else if(Arithmetic_Queue.size() > 0){
//         for(MatrixLane* lane : matrix_engine->matrix_lanes){
//             if(lane.isOccupied())
//             {
//                 grant = true;
//                 issueArithReq = true;
//                 break;
//             }
//         }
//         return grant;
//     }
//     return false;
// }

void MatrixDispatcher::startTicking()
{
    start();
    isBusy = true;
    DPRINTF(MatrixDispatcher, "Matrix Dispatcher has been started!\n");
}

void MatrixDispatcher::stopTicking()
{
    stop();
    isBusy = false;
    DPRINTF(MatrixDispatcher, "Matrix Dispatcher has been stopped!\n");
}

bool MatrixDispatcher::isOccupied()
{
    return isBusy;
}
void MatrixDispatcher::evaluate()
{
    // bool free_prf_avaliable = !matrix_engine->matrix_rename->freeList_empty();

    // if(matrix_engine->ew_unit->isIdle() == true){
    //     matrix_engine->ew_unit->startTicking();
    // }

    bool LaneisWorking = false;
    testnum++;
    for(uint8_t j = 0; j < matrix_engine->lane_num; j++){
        if(matrix_engine->matrix_lanes[j]->isOccupied() == false){
            LaneisFree++;
            break;
        }
    }

    for(uint8_t j = 0; j < matrix_engine->lane_num; j++){
        if(matrix_engine->matrix_lanes[j]->isOccupied() == true){
            LaneisWorking = true;
            break;
        }
    }
    if(!LaneisWorking){
        allLaneisFree++;
    }

    if(!matrix_engine->matrix_rename->freeTileList_empty()){
        hasFreeTileReg++;
    }
    if(!matrix_engine->matrix_rename->freeAccList_empty()){
        hasFreeAccReg++;
    }
    // if(!free_prf_avaliable){
    //     WaitingForReg++;
    // }
    //update size!
    //Delete in 24/7/9 for cancel the config stall
    // sizeM = matrix_engine->get_sizeM();
    // sizeN = matrix_engine->get_sizeN();
    // sizeK = matrix_engine->get_sizeK();
    //when meet with config instruction
    // if(config_stall){
    // if((Memory_Queue.size()==0)&&Arithmetic_Queue.size()==0){
    // if(cfg_Queue.size()!=0){
    //     ScoreBoard_Entry* matrix_sbe = cfg_Queue.back();
    //     if(matrix_sbe->_minst->ismcfgki()){
    //         matrix_engine->cfgSizeK(matrix_sbe->_minst->uimm10()); 
    //     }
    //     if(matrix_sbe->_minst->ismcfgmi()){
    //         matrix_engine->cfgSizeM(matrix_sbe->_minst->uimm10());
    //     }
    //     if(matrix_sbe->_minst->ismcfgni()){
    //         matrix_engine->cfgSizeN(matrix_sbe->_minst->uimm10());
    //     }
    //     if(matrix_sbe->_minst->ismcfgk()){
    //         matrix_engine->cfgSizeK(matrix_sbe->get_rs1_value());
    //     }
    //     if(matrix_sbe->_minst->ismcfgm()){
    //         matrix_engine->cfgSizeM(matrix_sbe->get_rs1_value());
    //     }
    //     if(matrix_sbe->_minst->ismcfgn()){
    //         matrix_engine->cfgSizeN(matrix_sbe->get_rs1_value());
    //     }
    //     if(matrix_sbe->_minst->ismcfg()){
    //         matrix_engine->cfgSizeN(matrix_sbe->_minst->getbits_8_15());
    //         matrix_engine->cfgSizeM(matrix_sbe->_minst->getbits_0_7());
    //         matrix_engine->cfgSizeK(matrix_sbe->_minst->getbits_16_31());
    //     }

    //     cfg_Queue.pop_back();
    //         // delete matrix_sbe;
    //         // config_stall = false;
    //     }
    //     }
    // }
    if(Memory_Queue.size()>MatrixMQentryUsed.value()){
        MatrixMQentryUsed = Memory_Queue.size();
    }
    if(Arithmetic_Queue.size()>MatrixAQentryUsed.value()){
        MatrixAQentryUsed = Arithmetic_Queue.size();
    }
    // if((Memory_Queue.size()==0)&&(Arithmetic_Queue.size()==0)&&!config_stall)
    if((Memory_Queue.size()==0)&&(Arithmetic_Queue.size()==0))
    {
        stopTicking();
        DPRINTF(MatrixDispatcher, "Matrix Dispatcher has been stoped, because no inst here!\n");
    }

    if(Memory_Queue.size()!=0 && !matrix_engine->matrix_mmu->isFull())
    {
        // MemQueueEntry* mentry;
        // uint32_t issue_range = (OoO) ? MQ_depth : 1;
        // uint32_t issue_range = (OoO) ? Memory_Queue.size() : 1;
        // for (uint32_t i = 0; i < issue_range; i++)
    for (auto mentry = Memory_Queue.begin(); mentry != Memory_Queue.end();)
{
    auto index = std::distance(Memory_Queue.begin(), mentry);
    bool ready_to_issue = false;
    // mentry = Memory_Queue[i];
    if((*mentry)->matrix_sbe._minst->isLoad() && !(*mentry)->matrix_sbe.isSent){
        //Because this instruction can be here, must be a free reg alias to it. So all operator are avaliable. 
        ready_to_issue = true;
    } else if((*mentry)->matrix_sbe._minst->isStore() && !(*mentry)->matrix_sbe.isSent){
        // ready_to_issue = matrix_engine->matrix_rename->get_RAT_vld(mentry->matrix_sbe.get_renamed_src3());
        ready_to_issue = matrix_engine->matrix_rename->get_PR_vld((*mentry)->matrix_sbe.get_renamed_src3()) && matrix_engine->matrix_rob->raw_solved((*mentry)->matrix_sbe.ms3, (*mentry)->matrix_sbe.src3_memdep_idx);
        MQread++;
    } else if ((*mentry)->matrix_sbe._minst->ismloadspm() && !(*mentry)->matrix_sbe.isSent){
        ready_to_issue = true;
    } else if ((*mentry)->matrix_sbe._minst->ismstorespm() && !(*mentry)->matrix_sbe.isSent){
        ready_to_issue = matrix_engine->matrix_rename->get_PR_vld((*mentry)->matrix_sbe.get_renamed_src3()) && matrix_engine->matrix_rob->raw_solved((*mentry)->matrix_sbe.ms3, (*mentry)->matrix_sbe.src3_memdep_idx);
    } else if((*mentry)->matrix_sbe._minst->ismmov_mm() && !(*mentry)->matrix_sbe.isSent) {
        ready_to_issue = matrix_engine->matrix_rename->get_PR_vld((*mentry)->matrix_sbe.get_renamed_src1()) && matrix_engine->matrix_rob->raw_solved((*mentry)->matrix_sbe.ms1, (*mentry)->matrix_sbe.src1_memdep_idx);
    } else if(((*mentry)->matrix_sbe._minst->ismmovb_m_x() || (*mentry)->matrix_sbe._minst->ismmovw_m_x()) && !(*mentry)->matrix_sbe.isSent) {
        ready_to_issue = true;
    } else if(((*mentry)->matrix_sbe._minst->ismmovb_x_m() || (*mentry)->matrix_sbe._minst->ismmovw_x_m()) && !(*mentry)->matrix_sbe.isSent) {
        ready_to_issue = matrix_engine->matrix_rename->get_PR_vld((*mentry)->matrix_sbe.get_renamed_src2() && matrix_engine->matrix_rob->raw_solved((*mentry)->matrix_sbe.ms2, (*mentry)->matrix_sbe.src2_memdep_idx));
    } else if((*mentry)->matrix_sbe._minst->ismzero() && !(*mentry)->matrix_sbe.isSent) {
        ready_to_issue = true;
    }

    if((*mentry)->matrix_sbe._minst->ismlae8() || (*mentry)->matrix_sbe._minst->ismlae8_spm() || (*mentry)->matrix_sbe._minst->ismsae8() || (*mentry)->matrix_sbe._minst->ismsae8_spm()) {
        (*mentry)->matrix_sbe.matrix_type = 0;
        (*mentry)->matrix_sbe.istranspose = false;
        (*mentry)->matrix_sbe.set_rows((*mentry)->matrix_sbe.get_cfg_sizeM());
        (*mentry)->matrix_sbe.set_data_size((*mentry)->matrix_sbe.get_cfg_sizeK());
    }
    else if((*mentry)->matrix_sbe._minst->ismlate8() || (*mentry)->matrix_sbe._minst->ismlate8_spm() || (*mentry)->matrix_sbe._minst->ismsate8() || (*mentry)->matrix_sbe._minst->ismsate8_spm()) {
        (*mentry)->matrix_sbe.matrix_type = 0;
        (*mentry)->matrix_sbe.istranspose = true;
        (*mentry)->matrix_sbe.set_rows((*mentry)->matrix_sbe.get_cfg_sizeK());
        (*mentry)->matrix_sbe.set_data_size((*mentry)->matrix_sbe.get_cfg_sizeM());
    }
    else if((*mentry)->matrix_sbe._minst->ismlbe8() || (*mentry)->matrix_sbe._minst->ismsbe8()) {
        (*mentry)->matrix_sbe.matrix_type = 1;
        (*mentry)->matrix_sbe.istranspose = false;
        (*mentry)->matrix_sbe.set_rows((*mentry)->matrix_sbe.get_cfg_sizeN());
        (*mentry)->matrix_sbe.set_data_size((*mentry)->matrix_sbe.get_cfg_sizeK());
    }
    else if((*mentry)->matrix_sbe._minst->ismlbe8_spm() || (*mentry)->matrix_sbe._minst->ismsbe8_spm()) {
        (*mentry)->matrix_sbe.matrix_type = 1;
        (*mentry)->matrix_sbe.istranspose = false;
        (*mentry)->matrix_sbe.set_rows((*mentry)->matrix_sbe.get_cfg_sizeK() / 4);
        (*mentry)->matrix_sbe.set_data_size((*mentry)->matrix_sbe.get_cfg_sizeN());
    }
    else if((*mentry)->matrix_sbe._minst->ismlbte8() || (*mentry)->matrix_sbe._minst->ismsbte8()) {
        (*mentry)->matrix_sbe.matrix_type = 1;
        (*mentry)->matrix_sbe.istranspose = true;
        (*mentry)->matrix_sbe.set_rows((*mentry)->matrix_sbe.get_cfg_sizeK());
        (*mentry)->matrix_sbe.set_data_size((*mentry)->matrix_sbe.get_cfg_sizeN());
    }
    else if((*mentry)->matrix_sbe._minst->ismlbte8_spm() || (*mentry)->matrix_sbe._minst->ismsbte8_spm()) {
        (*mentry)->matrix_sbe.matrix_type = 1;
        (*mentry)->matrix_sbe.istranspose = true;
        (*mentry)->matrix_sbe.set_rows((*mentry)->matrix_sbe.get_cfg_sizeN());
        (*mentry)->matrix_sbe.set_data_size((*mentry)->matrix_sbe.get_cfg_sizeK());
    }
    else if((*mentry)->matrix_sbe._minst->ismlce32() || (*mentry)->matrix_sbe._minst->ismlce32_spm() || (*mentry)->matrix_sbe._minst->ismsce32() || (*mentry)->matrix_sbe._minst->ismsce32_spm()) {
        (*mentry)->matrix_sbe.matrix_type = 2;
        (*mentry)->matrix_sbe.istranspose = false;
        (*mentry)->matrix_sbe.set_rows((*mentry)->matrix_sbe.get_cfg_sizeM());
        (*mentry)->matrix_sbe.set_data_size((*mentry)->matrix_sbe.get_cfg_sizeN() * 4);
    }
    else if((*mentry)->matrix_sbe._minst->ismlcte32() || (*mentry)->matrix_sbe._minst->ismlcte32_spm() || (*mentry)->matrix_sbe._minst->ismscte32() || (*mentry)->matrix_sbe._minst->ismscte32_spm()) {
        (*mentry)->matrix_sbe.matrix_type = 2;
        (*mentry)->matrix_sbe.istranspose = true;
        (*mentry)->matrix_sbe.set_rows((*mentry)->matrix_sbe.get_cfg_sizeM());
        (*mentry)->matrix_sbe.set_data_size((*mentry)->matrix_sbe.get_cfg_sizeN() * 4);
    }
    else if((*mentry)->matrix_sbe._minst->ismmov_mm()) {
        (*mentry)->matrix_sbe.matrix_type = 2;
        (*mentry)->matrix_sbe.istranspose = false;
        (*mentry)->matrix_sbe.set_rows((*mentry)->matrix_sbe.get_cfg_sizeM());
        (*mentry)->matrix_sbe.set_data_size((*mentry)->matrix_sbe.get_cfg_sizeN() * 4);
    }

    if(ready_to_issue || (*mentry)->matrix_sbe.isSent){
        IssueNum++;
        // DPRINTF(MatrixDispatcher, "memory queue idx %u is sent!\n", index);
        (*mentry)->matrix_sbe.isSent = true;
        bool success = false;
        uint64_t addr_index;
        addr_index = (*mentry)->matrix_sbe.get_rs1_value() + ((*mentry)->matrix_sbe.get_rs2_value())*((*mentry)->matrix_sbe.get_executed_num());
        // uint32_t spm_A_addr_index = (*mentry)->matrix_sbe.get_spm_addr() + 4 * (((*mentry)->matrix_sbe.get_executed_num()) * ((*mentry)->matrix_sbe.get_cfg_sizeM()));
        // uint32_t spm_B_addr_index = (*mentry)->matrix_sbe.get_spm_addr() + 4 * (((*mentry)->matrix_sbe.get_executed_num()) * ((*mentry)->matrix_sbe.get_cfg_sizeN()));
        uint32_t spm_addr_index = (*mentry)->matrix_sbe.get_spm_addr() + (((*mentry)->matrix_sbe.get_executed_num()) * ((*mentry)->matrix_sbe.get_stride()));

        if((*mentry)->matrix_sbe._minst->ismloadspm() && !(*mentry)->matrix_sbe.isSendAndWait){
            success = matrix_engine->matrix_spm->isAvailable((*mentry)->matrix_sbe.get_setid(), (*mentry)->matrix_sbe.get_rows() - (*mentry)->matrix_sbe.get_load_cnt() - 1, true, (*mentry)->matrix_sbe.spmReq->fence, (*mentry)->matrix_sbe.spm_reqId);
            if(success){
                // matrix_engine->MEreadSPM((*mentry)->matrix_sbe.get_setid() == 0 ? spm_A_addr_index : spm_B_addr_index, (*mentry)->matrix_sbe.get_stride() , static_cast<AccessMode>((*mentry)->matrix_sbe.get_mode()),
                if((*mentry)->matrix_sbe.get_dst_lrf_num() < 4 && (*mentry)->matrix_sbe.get_setid() == 1) {
                    spm_addr_index = (*mentry)->matrix_sbe.get_spm_addr() + (((*mentry)->matrix_sbe.get_executed_num() * 4) * ((*mentry)->matrix_sbe.get_stride()));
                    for(int i = 0; i < 4; i++) {
                        matrix_engine->MEreadSPM(spm_addr_index + i * (*mentry)->matrix_sbe.get_stride(), (*mentry)->matrix_sbe.get_row_size(), static_cast<AccessMode>((*mentry)->matrix_sbe.get_mode()),
                        [this, matrix_sbe_copy = (*(*mentry)).matrix_sbe, mentry_copy = (*mentry), i](uint8_t *data, uint8_t size){
                            this->matrix_engine->matrix_reg->wt_Brow(matrix_sbe_copy.get_dst_prf_num(), i, matrix_sbe_copy.get_executed_num(), data, size);
                            this->matrix_engine->matrix_reg->rls(matrix_sbe_copy.get_dst_prf_num());
                            if(i == 3)
                                mentry_copy->matrix_sbe.load();
                        });
                    }
                }
                else {
                    matrix_engine->MEreadSPM(spm_addr_index, (*mentry)->matrix_sbe.get_row_size(), static_cast<AccessMode>((*mentry)->matrix_sbe.get_mode()),
                    [this, matrix_sbe_copy = (*(*mentry)).matrix_sbe, mentry_copy = (*mentry)](uint8_t *data, uint8_t size){
                        if(matrix_sbe_copy.get_dst_lrf_num() < 4){
                            if(matrix_sbe_copy.get_setid() == 0){
                                this->matrix_engine->matrix_reg->wt_A_4col(matrix_sbe_copy.get_dst_prf_num(), 0, matrix_sbe_copy.get_executed_num(), data, size);
                            } else if (matrix_sbe_copy.get_setid() == 1){
                                this->matrix_engine->matrix_reg->wt_Brow(matrix_sbe_copy.get_dst_prf_num(), matrix_sbe_copy.get_executed_num() % 4, matrix_sbe_copy.get_executed_num() / 4, data, size);
                            } else{
                                panic("SPM LOAD setid error!\n");
                            }
                        } else if (matrix_sbe_copy.get_dst_lrf_num() < 8 && matrix_sbe_copy.get_dst_lrf_num() >= 4){
                            //!!!!!!!!!!!!!!!    //!!这里data是32的。但是输入是8!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                            if(matrix_sbe_copy.get_setid() == 2){
                                if(matrix_sbe_copy.istranspose) {
                                    this->matrix_engine->matrix_reg->wt_Ctrow(matrix_sbe_copy.get_dst_prf_num(), 0, matrix_sbe_copy.get_executed_num(), 0, data, size);
                                }
                                else {
                                    this->matrix_engine->matrix_reg->wt_Crow(matrix_sbe_copy.get_dst_prf_num(), 0, matrix_sbe_copy.get_executed_num(), 0, data, size);
                                }
                            } else {
                                panic("SPM LOAD setid error for acc!\n");
                            }
                        }

                        this->matrix_engine->matrix_reg->rls(matrix_sbe_copy.get_dst_prf_num());
                        mentry_copy->matrix_sbe.load();
                    });
                }
                
                (*mentry)->matrix_sbe.execute();
                success = false; //reset success signal
                matrix_engine->matrix_rename->set_PR_vld((*mentry)->matrix_sbe.get_dst_prf_num(), false);
                if((*mentry)->matrix_sbe.get_setid() == 0 || (*mentry)->matrix_sbe.get_setid() == 1 || (*mentry)->matrix_sbe.get_setid() == 2){
                    if ((*mentry)->matrix_sbe.get_executed_num() == (*mentry)->matrix_sbe.get_rows()){
                        (*mentry)->matrix_sbe.isSendAndWait = true;
                    }
                    break;
                } else {
                    panic("SPM LOAD setid error!\n");
                }
            }
        }

        else if((*mentry)->matrix_sbe._minst->ismstorespm() && !(*mentry)->matrix_sbe.isSendAndWait){
            
            uint8_t* mreg_row_data;
            if((*mentry)->matrix_sbe.get_setid() == 0){
                mreg_row_data = matrix_engine->matrix_reg->rd_A_4col((*mentry)->matrix_sbe.get_renamed_src3(), 0, (*mentry)->matrix_sbe.get_executed_num(), 32);// 32给大了可能要改
            } else if((*mentry)->matrix_sbe.get_setid() == 1){
                mreg_row_data = matrix_engine->matrix_reg->rd_Brow((*mentry)->matrix_sbe.get_renamed_src3(), 0, (*mentry)->matrix_sbe.get_executed_num(), 8);
            } else if((*mentry)->matrix_sbe.get_setid() == 2){
                if((*mentry)->matrix_sbe.istranspose)
                    mreg_row_data = matrix_engine->matrix_reg->rd_Ctrow((*mentry)->matrix_sbe.get_renamed_src3(), 0, (*mentry)->matrix_sbe.get_executed_num(), 0, (*mentry)->matrix_sbe.get_row_size());
                else
                    mreg_row_data = matrix_engine->matrix_reg->rd_Crow((*mentry)->matrix_sbe.get_renamed_src3(), 0, (*mentry)->matrix_sbe.get_executed_num(), 0, (*mentry)->matrix_sbe.get_row_size());
            } else{
                panic("SPM STORE setid error!\n");
            }
            // delete[] mreg_row_data;
            success = matrix_engine->matrix_spm->isAvailable((*mentry)->matrix_sbe.get_setid(), (*mentry)->matrix_sbe.get_rows() - (*mentry)->matrix_sbe.get_store_cnt() - 1, false, (*mentry)->matrix_sbe.spmReq->fence, (*mentry)->matrix_sbe.spm_reqId);
            if(success){
                // matrix_engine->MEwriteSPM((*mentry)->matrix_sbe.get_setid() == 0 ? spm_A_addr_index : spm_B_addr_index, mreg_row_data, (*mentry)->matrix_sbe.get_row_size(), static_cast<AccessMode>((*mentry)->matrix_sbe.get_mode()),
                matrix_engine->MEwriteSPM(spm_addr_index, mreg_row_data, (*mentry)->matrix_sbe.get_row_size(), static_cast<AccessMode>((*mentry)->matrix_sbe.get_mode()),
                [this, matrix_sbe_copy = (*(*mentry)).matrix_sbe, mentry_copy = (*mentry), mreg_row_data](){
                    this->matrix_engine->matrix_reg->rls(matrix_sbe_copy.get_renamed_src3());
                    mentry_copy->matrix_sbe.store();
                    delete[] mreg_row_data;
                });

                (*mentry)->matrix_sbe.execute();
                success = false; //reset success signal

                if((*mentry)->matrix_sbe.get_setid() == 0 || (*mentry)->matrix_sbe.get_setid() == 1 || (*mentry)->matrix_sbe.get_setid() == 2){
                    if ((*mentry)->matrix_sbe.get_executed_num() == (*mentry)->matrix_sbe.get_rows()){
                        (*mentry)->matrix_sbe.isSendAndWait = true;
                    }
                    break;
                } else {
                    panic("SPM STORE setid error!\n");
                }
            } else {
                delete[] mreg_row_data;
            }
        }

        else if((*mentry)->matrix_sbe._minst->isLoad() && !(*mentry)->matrix_sbe.isSendAndWait){
            // // occupy write port
            // if(!matrix_engine->matrix_reg->occupy_wtport()){
            //     ++mentry;
            //     continue;
            // }
            //Load MB once a row!
            success = matrix_engine->matrix_mmu->readMatrixMem(addr_index, (*mentry)->matrix_sbe.get_data_size(), &((*mentry)->tc), 0, (*mentry)->matrix_sbe.get_dst_prf_num(), [this, matrix_sbe_copy = (*(*mentry)).matrix_sbe, mentry_copy = (*mentry)](uint8_t *data, uint8_t size){
                if(matrix_sbe_copy.get_dst_lrf_num() < 4){
                    if(matrix_sbe_copy.istranspose)
                    {
                        for(uint8_t i = 0; i < matrix_sbe_copy.get_data_size(); i++){
                            this->matrix_engine->matrix_reg->wtreg_byte(matrix_sbe_copy.get_dst_prf_num(), matrix_sbe_copy.get_executed_num() % (this->matrix_engine->matrix_reg->bank_num), matrix_sbe_copy.get_executed_num() / (this->matrix_engine->matrix_reg->bank_num), i, data[i]);
                        }
                    }
                    else {
                        for(uint8_t i = 0; i < matrix_sbe_copy.get_data_size(); i++){
                            this->matrix_engine->matrix_reg->wtreg_byte(matrix_sbe_copy.get_dst_prf_num(), i % (this->matrix_engine->matrix_reg->bank_num), i / (this->matrix_engine->matrix_reg->bank_num), matrix_sbe_copy.get_executed_num(), data[i]);
                        }
                    }
                    
                } else if (matrix_sbe_copy.get_dst_lrf_num() < 8 && matrix_sbe_copy.get_dst_lrf_num() >= 4){
                    if(matrix_sbe_copy.istranspose)
                    {
                        for(uint8_t i = 0; i < (matrix_sbe_copy.get_data_size() / 4); i++){
                            uint32_t data_sp = static_cast<uint32_t>(data[4*i]) | (static_cast<uint32_t>(data[4*i+1]) << 8) | (static_cast<uint32_t>(data[4*i+2]) << 16) | (static_cast<uint32_t>(data[4*i+3]) << 24);
                            this->matrix_engine->matrix_reg->wtreg_int32(matrix_sbe_copy.get_dst_prf_num(), matrix_sbe_copy.get_executed_num() / 2, i, matrix_sbe_copy.get_executed_num()%2, data_sp);
                        }
                    }
                    else {
                        for(uint8_t i = 0; i < (matrix_sbe_copy.get_data_size() / 4); i++){
                            uint32_t data_sp = static_cast<uint32_t>(data[4*i]) | (static_cast<uint32_t>(data[4*i+1]) << 8) | (static_cast<uint32_t>(data[4*i+2]) << 16) | (static_cast<uint32_t>(data[4*i+3]) << 24);
                            this->matrix_engine->matrix_reg->wtreg_int32(matrix_sbe_copy.get_dst_prf_num(), i / 2, matrix_sbe_copy.get_executed_num(), i%2, data_sp);
                        }
                    }
                }
                this->matrix_engine->matrix_reg->rls(matrix_sbe_copy.get_dst_prf_num());
                if(data[31] == 0){
                    DPRINTF(MatrixDispatcher, "find zero in loading stage!!\n");
                }
                mentry_copy->matrix_sbe.load();
            });
            if(success){
                (*mentry)->matrix_sbe.execute();
                success = false; //reset success signal
                matrix_engine->matrix_rename->set_PR_vld((*mentry)->matrix_sbe.get_dst_prf_num(), false);
                // DPRINTF(MatrixDispatcher, "pr idx %d is valid now\n", (*mentry)->matrix_sbe.get_dst_prf_num());
                if ((*mentry)->matrix_sbe.get_executed_num() == (*mentry)->matrix_sbe.get_rows()){
                    (*mentry)->matrix_sbe.isSendAndWait = true;
                }
                break;
            }
        } else if((*mentry)->matrix_sbe._minst->isStore() && !(*mentry)->matrix_sbe.isSendAndWait){
            //Store MB once a row!
            if(!matrix_engine->matrix_reg->try_occupy((*mentry)->matrix_sbe.get_renamed_src3())){
                ++mentry;
                continue;
            }
            uint8_t* mreg_row_data;
            if((*mentry)->matrix_sbe.get_renamed_src3() < tileReg_num){
                mreg_row_data = new uint8_t[(*mentry)->matrix_sbe.get_data_size()];
                if((*mentry)->matrix_sbe.istranspose)
                {
                    for(uint32_t j = 0; j < (*mentry)->matrix_sbe.get_data_size(); j++){
                        mreg_row_data[j] = matrix_engine->matrix_reg->rdreg_byte((*mentry)->matrix_sbe.get_renamed_src3(), (*mentry)->matrix_sbe.get_executed_num() % matrix_engine->matrix_reg->bank_num, (*mentry)->matrix_sbe.get_executed_num() / matrix_engine->matrix_reg->bank_num, j);
                    }
                }
                else {
                    for(uint32_t j = 0; j < (*mentry)->matrix_sbe.get_data_size(); j++){
                        mreg_row_data[j] = matrix_engine->matrix_reg->rdreg_byte((*mentry)->matrix_sbe.get_renamed_src3(), j % matrix_engine->matrix_reg->bank_num, j / matrix_engine->matrix_reg->bank_num, (*mentry)->matrix_sbe.get_executed_num());
                    }
                }
            } else if ((*mentry)->matrix_sbe.get_renamed_src3() < tileReg_num + accReg_num && (*mentry)->matrix_sbe.get_renamed_src3() >= tileReg_num){
                mreg_row_data = new uint8_t[(*mentry)->matrix_sbe.get_data_size()];
                if((*mentry)->matrix_sbe.istranspose)
                {
                    for(uint32_t j = 0; j < ((*mentry)->matrix_sbe.get_data_size() / 4); j++){
                        uint32_t data_sp = matrix_engine->matrix_reg->rdreg_int32((*mentry)->matrix_sbe.get_renamed_src3(), (*mentry)->matrix_sbe.get_executed_num() / 2, j, (*mentry)->matrix_sbe.get_executed_num() % 2);
                        mreg_row_data[4*j] = static_cast<uint8_t>(data_sp & 0xFF);
                        mreg_row_data[4*j+1] = static_cast<uint8_t>((data_sp >> 8) & 0xFF); 
                        mreg_row_data[4*j+2] = static_cast<uint8_t>((data_sp >> 16) & 0xFF);
                        mreg_row_data[4*j+3] = static_cast<uint8_t>((data_sp >> 24) & 0xFF);
                    }
                }
                else {
                    for(uint32_t j = 0; j < ((*mentry)->matrix_sbe.get_data_size() / 4); j++){
                        uint32_t data_sp = matrix_engine->matrix_reg->rdreg_int32((*mentry)->matrix_sbe.get_renamed_src3(), j / 2, (*mentry)->matrix_sbe.get_executed_num(), j % 2);
                        mreg_row_data[4*j] = static_cast<uint8_t>(data_sp & 0xFF);
                        mreg_row_data[4*j+1] = static_cast<uint8_t>((data_sp >> 8) & 0xFF); 
                        mreg_row_data[4*j+2] = static_cast<uint8_t>((data_sp >> 16) & 0xFF);
                        mreg_row_data[4*j+3] = static_cast<uint8_t>((data_sp >> 24) & 0xFF);
                    }
                }
            }
            success = matrix_engine->matrix_mmu->writeMatrixMem(addr_index, mreg_row_data, (*mentry)->matrix_sbe.get_data_size(), &((*mentry)->tc), 0, [this, matrix_sbe_copy = (*(*mentry)).matrix_sbe, mentry_copy = (*mentry), mreg_row_data](){
                this->matrix_engine->matrix_reg->rls(matrix_sbe_copy.get_renamed_src3());
                mentry_copy->matrix_sbe.store();
                delete[] mreg_row_data; // move this here to ensure it is deleted after store operation
            });
            if(success){
                (*mentry)->matrix_sbe.execute();
                success = false; // reset success signal
                if ((*mentry)->matrix_sbe.get_executed_num() == (*mentry)->matrix_sbe.get_rows()){
                    (*mentry)->matrix_sbe.isSendAndWait = true;
                }
                break;
            }
        }

        else if((*mentry)->matrix_sbe._minst->ismmov_mm() && !(*mentry)->matrix_sbe.isSendAndWait) {
            if(!matrix_engine->matrix_reg->try_occupy((*mentry)->matrix_sbe.get_renamed_src1()) || !matrix_engine->matrix_reg->try_occupy((*mentry)->matrix_sbe.get_dst_prf_num())){
                ++mentry;
                continue;
            }
            uint32_t* mreg_row_data = new uint32_t[(*mentry)->matrix_sbe.get_data_size()];
            if((*mentry)->matrix_sbe.get_renamed_src1() < tileReg_num) {
                for(uint32_t j = 0; j < (*mentry)->matrix_sbe.get_data_size(); j++) {
                    uint8_t data_sp = matrix_engine->matrix_reg->rdreg_byte((*mentry)->matrix_sbe.get_renamed_src1(), j % matrix_engine->matrix_reg->bank_num, j / matrix_engine->matrix_reg->bank_num, (*mentry)->matrix_sbe.get_executed_num());
                    mreg_row_data[j] = static_cast<uint32_t>(data_sp);
                }
            }
            else if((*mentry)->matrix_sbe.get_renamed_src1() < tileReg_num + accReg_num && (*mentry)->matrix_sbe.get_renamed_src1() >= tileReg_num) {
                for(uint32_t j = 0; j < (*mentry)->matrix_sbe.get_data_size(); j++) {
                    if(j < (*mentry)->matrix_sbe.get_data_size() / 4)
                        mreg_row_data[j] = matrix_engine->matrix_reg->rdreg_int32((*mentry)->matrix_sbe.get_renamed_src1(), j / 2, (*mentry)->matrix_sbe.get_executed_num(), j % 2);
                    else
                        mreg_row_data[j] = 0;
                }
            }
            if((*mentry)->matrix_sbe.get_dst_prf_num() < tileReg_num) {
                for(uint32_t i = 0; i < (*mentry)->matrix_sbe.get_data_size(); i++) {
                    this->matrix_engine->matrix_reg->wtreg_byte((*mentry)->matrix_sbe.get_dst_prf_num(), i % (this->matrix_engine->matrix_reg->bank_num), i / (this->matrix_engine->matrix_reg->bank_num), (*mentry)->matrix_sbe.get_executed_num(), static_cast<uint8_t>(mreg_row_data[i]));
                }
            }
            else if((*mentry)->matrix_sbe.get_dst_prf_num() < tileReg_num + accReg_num && (*mentry)->matrix_sbe.get_dst_prf_num() >= tileReg_num) {
                for(uint32_t i = 0; i < (*mentry)->matrix_sbe.get_data_size() / 4; i++) {
                    this->matrix_engine->matrix_reg->wtreg_int32((*mentry)->matrix_sbe.get_dst_prf_num(), i / 2, (*mentry)->matrix_sbe.get_executed_num(), i%2, mreg_row_data[i]);
                }
            }
            (*mentry)->matrix_sbe.execute();
            matrix_engine->matrix_rename->set_PR_vld((*mentry)->matrix_sbe.get_dst_prf_num(), false);
            // DPRINTF(MatrixDispatcher, "pr idx %d is valid now\n", (*mentry)->matrix_sbe.get_dst_prf_num());
            if ((*mentry)->matrix_sbe.get_executed_num() == (*mentry)->matrix_sbe.get_rows()){
                (*mentry)->matrix_sbe.isSendAndWait = true;
            }
            break;
            delete[] mreg_row_data;
        }

        else if(((*mentry)->matrix_sbe._minst->ismmovb_m_x() || (*mentry)->matrix_sbe._minst->ismmovw_m_x()) && !(*mentry)->matrix_sbe.isSendAndWait) {
            if((*mentry)->matrix_sbe.get_dst_prf_num() < tileReg_num) {
                this->matrix_engine->matrix_reg->wtreg_byte((*mentry)->matrix_sbe.get_dst_prf_num(), 
                    (*mentry)->matrix_sbe.get_rs1_value() % (*mentry)->matrix_sbe.get_cfg_sizeK() % (this->matrix_engine->matrix_reg->bank_num), 
                    (*mentry)->matrix_sbe.get_rs1_value() % (*mentry)->matrix_sbe.get_cfg_sizeK() / (this->matrix_engine->matrix_reg->bank_num), 
                    (*mentry)->matrix_sbe.get_rs1_value() / (*mentry)->matrix_sbe.get_cfg_sizeK(), static_cast<uint8_t>((*mentry)->matrix_sbe.get_rs2_value() % 256));
            }
            else if((*mentry)->matrix_sbe.get_dst_prf_num() < tileReg_num + accReg_num && (*mentry)->matrix_sbe.get_dst_prf_num() >= tileReg_num) {
                this->matrix_engine->matrix_reg->wtreg_int32((*mentry)->matrix_sbe.get_dst_prf_num(),
                    (*mentry)->matrix_sbe.get_rs1_value() % (*mentry)->matrix_sbe.get_cfg_sizeN() / 2,
                    (*mentry)->matrix_sbe.get_rs1_value() / (*mentry)->matrix_sbe.get_cfg_sizeN(),
                    (*mentry)->matrix_sbe.get_rs1_value() % (*mentry)->matrix_sbe.get_cfg_sizeN() % 2, (*mentry)->matrix_sbe.get_rs2_value());
            }
            (*mentry)->matrix_sbe.isSendAndWait = true;
        }
        else if(((*mentry)->matrix_sbe._minst->ismmovb_x_m() || (*mentry)->matrix_sbe._minst->ismmovw_x_m()) && !(*mentry)->matrix_sbe.isSendAndWait) {
            if((*mentry)->matrix_sbe.get_renamed_src2() < tileReg_num) {
                uint64_t data_sp = matrix_engine->matrix_reg->rdreg_byte((*mentry)->matrix_sbe.get_renamed_src2(),
                    (*mentry)->matrix_sbe.get_rs1_value() % (*mentry)->matrix_sbe.get_cfg_sizeK() % (this->matrix_engine->matrix_reg->bank_num),
                    (*mentry)->matrix_sbe.get_rs1_value() % (*mentry)->matrix_sbe.get_cfg_sizeK() / (this->matrix_engine->matrix_reg->bank_num),
                    (*mentry)->matrix_sbe.get_rs1_value() / (*mentry)->matrix_sbe.get_cfg_sizeK());
                (*mentry)->matrix_sbe.callDoneCallback(data_sp);
            }
            else if((*mentry)->matrix_sbe.get_renamed_src2() < tileReg_num + accReg_num && (*mentry)->matrix_sbe.get_renamed_src2() >= tileReg_num) {
                uint64_t data_sp = matrix_engine->matrix_reg->rdreg_int32((*mentry)->matrix_sbe.get_renamed_src2(),
                    (*mentry)->matrix_sbe.get_rs1_value() % (*mentry)->matrix_sbe.get_cfg_sizeN() / 2,
                    (*mentry)->matrix_sbe.get_rs1_value() / (*mentry)->matrix_sbe.get_cfg_sizeN(),
                    (*mentry)->matrix_sbe.get_rs1_value() % (*mentry)->matrix_sbe.get_cfg_sizeN() % 2);
                (*mentry)->matrix_sbe.callDoneCallback(data_sp);
            }
            (*mentry)->matrix_sbe.isSendAndWait = true;
        }
        else if((*mentry)->matrix_sbe._minst->ismzero() && !(*mentry)->matrix_sbe.isSendAndWait) {
            (*mentry)->matrix_sbe.isSendAndWait = true;
            matrix_engine->matrix_reg->set_reg_zero((*mentry)->matrix_sbe.get_dst_prf_num());
            matrix_engine->matrix_rename->set_PR_vld((*mentry)->matrix_sbe.get_dst_prf_num(), true);
            matrix_engine->matrix_rename->regrls((*mentry)->matrix_sbe.get_dst_prf_num());
            matrix_engine->matrix_rob->set_rob_entry_executed((*mentry)->matrix_sbe.get_rob_entry());
            matrix_engine->matrix_rob->set_md_entry_valid((*mentry)->matrix_sbe.md, (*mentry)->matrix_sbe.dst_memdep_idx);
            DPRINTF(MatrixDispatcher, "matrixzero set md entry valid, idx = %d, dst_memdep_idx = %d, dst = %d\n", (*mentry)->matrix_sbe.md, (*mentry)->matrix_sbe.dst_memdep_idx, (*mentry)->matrix_sbe.get_dst_prf_num());
            auto mentry_to_delete = *mentry;
            mentry = Memory_Queue.erase(mentry);
            if(mentry_to_delete != nullptr) {
                delete mentry_to_delete;
            }
            continue;
        }

        // DPRINTF(MatrixDispatcher, "(*mentry)->matrix_sbe.get_load_cnt() = %d, (*mentry)->matrix_sbe.get_store_cnt() = %d, (*mentry)->matrix_sbe.get_executed_num() = %d, (*mentry)->matrix_sbe.get_rows() = %d\n",
        //     (*mentry)->matrix_sbe.get_load_cnt(), (*mentry)->matrix_sbe.get_store_cnt(), (*mentry)->matrix_sbe.get_executed_num(), (*mentry)->matrix_sbe.get_rows());
        if((*mentry)->matrix_sbe.isSendAndWait && (*mentry)->matrix_sbe._minst->isLoad() && (*mentry)->matrix_sbe.get_load_cnt() == (*mentry)->matrix_sbe.get_rows()){
            // load commit
            matrix_engine->matrix_rename->set_PR_vld((*mentry)->matrix_sbe.get_dst_prf_num(), true);
            matrix_engine->matrix_rename->regrls((*mentry)->matrix_sbe.get_dst_prf_num());
            // matrix_engine->matrix_reg->rls((*mentry)->matrix_sbe.get_dst_prf_num());
            // matrix_engine->matrix_reg->rls_wrport();
            // DPRINTF(MatrixDispatcher, "MQ: pr idx %d is valid now\n", (*mentry)->matrix_sbe.get_dst_prf_num());
            matrix_engine->matrix_rob->set_rob_entry_executed((*mentry)->matrix_sbe.get_rob_entry());
            matrix_engine->matrix_rob->set_md_entry_valid((*mentry)->matrix_sbe.ms3, (*mentry)->matrix_sbe.dst_memdep_idx);
            DPRINTF(MatrixDispatcher, "load set md entry valid, idx = %d, dst_memdep_idx = %d\n", (*mentry)->matrix_sbe.ms3, (*mentry)->matrix_sbe.dst_memdep_idx);
            auto mentry_to_delete = *mentry;
            mentry = Memory_Queue.erase(mentry);
            if(mentry_to_delete != nullptr) {
                delete mentry_to_delete;
            }
            matrix_engine->matrix_rob->meminst_num--;
            // XY = false; //try to not depend on the config to reset the XY
        } else if((*mentry)->matrix_sbe.isSendAndWait && (*mentry)->matrix_sbe._minst->isStore() && (*mentry)->matrix_sbe.get_store_cnt() == (*mentry)->matrix_sbe.get_rows()) {
            // store commit
            // matrix_engine->matrix_reg->rls((*mentry)->matrix_sbe.get_dst_prf_num());
            // matrix_engine->matrix_rob->set_rob_entry_executed((*mentry)->matrix_sbe.get_rob_entry());
            // matrix_engine->matrix_rename->regrls((*mentry)->matrix_sbe.get_renamed_src3());
            matrix_engine->matrix_rename->readrls((*mentry)->matrix_sbe.get_renamed_src3(), (*mentry)->matrix_sbe.src3_read_idx);
            DPRINTF(MatrixDispatcherReadLock, "MQ: readrls src3 pr idx %d, read idx = %d\n", (*mentry)->matrix_sbe.get_renamed_src3(), (*mentry)->matrix_sbe.src3_read_idx);
            // std::cout << "Before erase, size: " << Memory_Queue.size() << std::endl;
            auto mentry_to_delete = *mentry;
            mentry = Memory_Queue.erase(mentry);
            if(mentry_to_delete != nullptr) {
                delete mentry_to_delete;
            }
            // std::cout << "After erase, size: " << Memory_Queue.size() << std::endl;
            matrix_engine->matrix_rob->meminst_num--;
        }
        // commit regspm
        else if((*mentry)->matrix_sbe.isSendAndWait && (*mentry)->matrix_sbe._minst->ismloadspm() && (*mentry)->matrix_sbe.get_load_cnt() == (*mentry)->matrix_sbe.get_rows()){
            matrix_engine->matrix_rename->set_PR_vld((*mentry)->matrix_sbe.get_dst_prf_num(), true);
            matrix_engine->matrix_rename->regrls((*mentry)->matrix_sbe.get_dst_prf_num());
            // matrix_engine->matrix_reg->rls((*mentry)->matrix_sbe.get_dst_prf_num());
            // matrix_engine->matrix_reg->rls_wrport();
            // DPRINTF(MatrixDispatcher, "MQ: pr idx %d is valid now\n", (*mentry)->matrix_sbe.get_dst_prf_num());
            matrix_engine->matrix_rob->set_rob_entry_executed((*mentry)->matrix_sbe.get_rob_entry());
            matrix_engine->matrix_rob->set_md_entry_valid((*mentry)->matrix_sbe.ms3, (*mentry)->matrix_sbe.dst_memdep_idx);
            if((*mentry)->matrix_sbe.get_setid() == 0)
                loadacommit++;
            else if((*mentry)->matrix_sbe.get_setid() == 1)
                loadbcommit++;
            else if((*mentry)->matrix_sbe.get_setid() == 2)
                loadccommit++;
            // DPRINTF(MatrixDispatcher, "mload a = %d, load b = %d, load c = %d, commited load a = %d, load b = %d, load c = %d\n", loada, loadb, loadc, loadacommit, loadbcommit, loadccommit);
            DPRINTF(MatrixDispatcher, "mload spm set md entry valid, idx = %d, dst_memdep_idx = %d\n", (*mentry)->matrix_sbe.ms3, (*mentry)->matrix_sbe.dst_memdep_idx);
            (*mentry)->matrix_sbe.spmReq->isExecuted = true;
            auto mentry_to_delete = *mentry;
            mentry = Memory_Queue.erase(mentry);
            if(mentry_to_delete != nullptr) {
                delete mentry_to_delete;
            }

            matrix_engine->matrix_rob->meminst_num--;
        } else if((*mentry)->matrix_sbe.isSendAndWait && (*mentry)->matrix_sbe._minst->ismstorespm() && (*mentry)->matrix_sbe.get_store_cnt() == (*mentry)->matrix_sbe.get_rows()){
            // matrix_engine->matrix_rename->regrls((*mentry)->matrix_sbe.get_renamed_src3());
            matrix_engine->matrix_rename->readrls((*mentry)->matrix_sbe.get_renamed_src3(), (*mentry)->matrix_sbe.src3_read_idx);
            DPRINTF(MatrixDispatcherReadLock, "MQ: readrls src3 pr idx %d, read idx = %d\n", (*mentry)->matrix_sbe.get_renamed_src3(), (*mentry)->matrix_sbe.src3_read_idx);
            // std::cout << "Before erase, size: " << Memory_Queue.size() << std::endl;
            (*mentry)->matrix_sbe.spmReq->isExecuted = true;
            auto mentry_to_delete = *mentry;
            mentry = Memory_Queue.erase(mentry);
            if(mentry_to_delete != nullptr) {
                delete mentry_to_delete;
            }
            // std::cout << "After erase, size: " << Memory_Queue.size() << std::endl;
            matrix_engine->matrix_rob->meminst_num--;
        }  else if((*mentry)->matrix_sbe.isSendAndWait && (*mentry)->matrix_sbe._minst->ismmov_mm()) {
            matrix_engine->matrix_rename->set_PR_vld((*mentry)->matrix_sbe.get_dst_prf_num(), true);
            matrix_engine->matrix_rename->regrls((*mentry)->matrix_sbe.get_dst_prf_num());
            matrix_engine->matrix_rename->regrls((*mentry)->matrix_sbe.get_renamed_src1());
            matrix_engine->matrix_rename->readrls((*mentry)->matrix_sbe.get_renamed_src1(), (*mentry)->matrix_sbe.src1_read_idx);
            // DPRINTF(MatrixDispatcher, "MQ: pr idx %d is valid now\n", (*mentry)->matrix_sbe.get_dst_prf_num());
            matrix_engine->matrix_rob->set_rob_entry_executed((*mentry)->matrix_sbe.get_rob_entry());
            matrix_engine->matrix_rob->set_md_entry_valid((*mentry)->matrix_sbe.md, (*mentry)->matrix_sbe.dst_memdep_idx);
            // std::cout << "Before erase, size: " << Memory_Queue.size() << std::endl;
            auto mentry_to_delete = *mentry;
            mentry = Memory_Queue.erase(mentry);
            if(mentry_to_delete != nullptr) {
                delete mentry_to_delete;
            }
            // std::cout << "After erase, size: " << Memory_Queue.size() << std::endl;
        }
        else if((*mentry)->matrix_sbe.isSendAndWait && ((*mentry)->matrix_sbe._minst->ismmovb_m_x() || (*mentry)->matrix_sbe._minst->ismmovw_m_x())) {
            matrix_engine->matrix_rename->set_PR_vld((*mentry)->matrix_sbe.get_dst_prf_num(), true);
            matrix_engine->matrix_rename->regrls((*mentry)->matrix_sbe.get_dst_prf_num());
            // DPRINTF(MatrixDispatcher, "MQ: pr idx %d is valid now\n", (*mentry)->matrix_sbe.get_dst_prf_num());
            matrix_engine->matrix_rob->set_rob_entry_executed((*mentry)->matrix_sbe.get_rob_entry());
            matrix_engine->matrix_rob->set_md_entry_valid((*mentry)->matrix_sbe.md, (*mentry)->matrix_sbe.dst_memdep_idx);
            // std::cout << "Before erase, size: " << Memory_Queue.size() << std::endl;
            auto mentry_to_delete = *mentry;
            mentry = Memory_Queue.erase(mentry);
            if(mentry_to_delete != nullptr) {
                delete mentry_to_delete;
            }
            // std::cout << "After erase, size: " << Memory_Queue.size() << std::endl;
        }  else if((*mentry)->matrix_sbe.isSendAndWait && ((*mentry)->matrix_sbe._minst->ismmovb_x_m() || (*mentry)->matrix_sbe._minst->ismmovw_x_m())) {
            matrix_engine->matrix_rename->regrls((*mentry)->matrix_sbe.get_renamed_src2());
            matrix_engine->matrix_rename->readrls((*mentry)->matrix_sbe.get_renamed_src2(), (*mentry)->matrix_sbe.src2_read_idx);
            // std::cout << "Before erase, size: " << Memory_Queue.size() << std::endl;
            auto mentry_to_delete = *mentry;
            mentry = Memory_Queue.erase(mentry);
            if(mentry_to_delete != nullptr) {
                delete mentry_to_delete;
            }
            // std::cout << "After erase, size: " << Memory_Queue.size() << std::endl;
        }
        else {
            ++mentry;
        }
    } else {
        ++mentry;
    }
}

    }

    if(Arithmetic_Queue.size()!=0)
    {
        //NOTE: NOW ONLY SUPPORT OoO ISSUE!!!
        // ArithQueueEntry* aentry;
        // uint32_t issue_range = (OoO) ? AQ_depth : 1;
        // uint32_t issue_range = (OoO) ? Arithmetic_Queue.size() : 1;
        // DPRINTF(MatrixDispatcher, "check aq!!\n");
        // for(uint32_t i = 0; i < issue_range; i++)
        bool freelane_waitinst = false;
        bool waitforlane = false;
        bool waitforewu = false;
        int cnt = 0;
        for(auto aentry = Arithmetic_Queue.begin(); aentry != Arithmetic_Queue.end();)
        {
            // aentry = Arithmetic_Queue[i];
            // ====
            //judge the inst status
            cnt++;
            // DPRINTF(MatrixDispatcher, "qentry number = %d\n", cnt);
            if((*aentry)->matrix_sbe.isIssue&&(!(*aentry)->matrix_sbe.isDone)){
                //just issued but have not done
                if((*aentry)->matrix_sbe._minst->ismmacc()){
                    if(matrix_engine->ew_unit->isDone((*aentry)->matrix_sbe.lane_idx) && matrix_engine->ew_unit->get_dest((*aentry)->matrix_sbe.lane_idx) == (*aentry)->matrix_sbe.get_dst_prf_num()){
                        (*aentry)->matrix_sbe.isDone = true;
                        matrix_engine->ew_unit->resetDone((*aentry)->matrix_sbe.lane_idx);
                    }
                } else if ((*aentry)->matrix_sbe._minst->ismadd()||(*aentry)->matrix_sbe._minst->ismsub()||(*aentry)->matrix_sbe._minst->ismmul()||(*aentry)->matrix_sbe._minst->ismmulh()||(*aentry)->matrix_sbe._minst->ismmax()||(*aentry)->matrix_sbe._minst->ismumax()||(*aentry)->matrix_sbe._minst->ismmin()||(*aentry)->matrix_sbe._minst->ismumin()||(*aentry)->matrix_sbe._minst->ismsll()||(*aentry)->matrix_sbe._minst->ismsrl()||(*aentry)->matrix_sbe._minst->ismsra()) {
                    if(matrix_engine->ew_unit->isDone((*aentry)->matrix_sbe.ewu_idx)){
                        (*aentry)->matrix_sbe.isDone = true;
                        matrix_engine->ew_unit->resetDone((*aentry)->matrix_sbe.lane_idx);
                    }
                } else if ((*aentry)->matrix_sbe._minst->ismredcadd() || (*aentry)->matrix_sbe._minst->ismredcmax()) {
                    if(matrix_engine->ew_unit->isDone((*aentry)->matrix_sbe.ewu_idx)){
                        (*aentry)->matrix_sbe.isDone = true;
                        matrix_engine->ew_unit->resetDone((*aentry)->matrix_sbe.lane_idx);
                    }
                }

                ++aentry;
            }else if((*aentry)->matrix_sbe.isDone){
                    // inst has been executed
                    matrix_engine->ew_unit->resetDone((*aentry)->matrix_sbe.ewu_idx);
                    matrix_engine->matrix_rename->set_PR_vld((*aentry)->matrix_sbe.get_dst_prf_num(), true);
                    matrix_engine->matrix_rename->regrls((*aentry)->matrix_sbe.get_dst_prf_num());
                    matrix_engine->matrix_rename->regrls((*aentry)->matrix_sbe.get_renamed_src1());
                    if((*aentry)->matrix_sbe._minst->ismmacc())
                        DPRINTF(MatrixDispatcher, "mac set md entry valid idx = %d, dst_memdep_idx = %d, pmd = %d\n", (*aentry)->matrix_sbe.md, (*aentry)->matrix_sbe.dst_memdep_idx, (*aentry)->matrix_sbe.get_dst_prf_num());
                    else
                        DPRINTF(MatrixDispatcher, "unit set md entry valid idx = %d, dst_memdep_idx = %d, pmd = %d\n", (*aentry)->matrix_sbe.md, (*aentry)->matrix_sbe.dst_memdep_idx, (*aentry)->matrix_sbe.get_dst_prf_num());
                    DPRINTF(MatrixDispatcher, " lane_idx = %d\n", (*aentry)->matrix_sbe.lane_idx);
                    if(!(*aentry)->matrix_sbe._minst->ismredcadd() && !(*aentry)->matrix_sbe._minst->ismredcmax() && !(*aentry)->matrix_sbe._minst->ismlut())
                    {
                        matrix_engine->matrix_rename->regrls((*aentry)->matrix_sbe.get_renamed_src2());
                    }
                    if((*aentry)->matrix_sbe._minst->isMatrixElementWise() && (((*aentry)->matrix_sbe.get_renamed_src1() != (*aentry)->matrix_sbe.get_dst_prf_num()) || ((*aentry)->matrix_sbe.get_renamed_src2() != (*aentry)->matrix_sbe.get_dst_prf_num()))) {
                        if((*aentry)->matrix_sbe.get_renamed_src1() != (*aentry)->matrix_sbe.get_dst_prf_num()) {
                            matrix_engine->matrix_rename->readrls((*aentry)->matrix_sbe.get_renamed_src1(), (*aentry)->matrix_sbe.src1_read_idx);
                            DPRINTF(MatrixDispatcherReadLock, "AQ: elementwise readrls src1 pr idx %d, read idx = %d\n", (*aentry)->matrix_sbe.get_renamed_src1(), (*aentry)->matrix_sbe.src1_read_idx);
                        }
                        if((*aentry)->matrix_sbe.get_renamed_src2() != (*aentry)->matrix_sbe.get_dst_prf_num()) {
                            matrix_engine->matrix_rename->readrls((*aentry)->matrix_sbe.get_renamed_src2(), (*aentry)->matrix_sbe.src2_read_idx);
                            DPRINTF(MatrixDispatcherReadLock, "AQ: elementwise readrls src2 pr idx %d, read idx = %d\n", (*aentry)->matrix_sbe.get_renamed_src2(), (*aentry)->matrix_sbe.src2_read_idx);
                        }
                    }
                    if(((*aentry)->matrix_sbe._minst->ismredcadd() || (*aentry)->matrix_sbe._minst->ismredcmax() || (*aentry)->matrix_sbe._minst->ismlut()) && (*aentry)->matrix_sbe.get_renamed_src1() != (*aentry)->matrix_sbe.get_dst_prf_num()) {
                        matrix_engine->matrix_rename->readrls((*aentry)->matrix_sbe.get_renamed_src1(), (*aentry)->matrix_sbe.src1_read_idx);
                        DPRINTF(MatrixDispatcherReadLock, "AQ: rec readrls src1 pr idx %d, read idx = %d\n", (*aentry)->matrix_sbe.get_renamed_src1(), (*aentry)->matrix_sbe.src1_read_idx);
                    }
                    // DPRINTF(MatrixDispatcher, "AQ: pr idx %d is valid now\n", (*aentry)->matrix_sbe.get_dst_prf_num());
                    matrix_engine->matrix_rob->set_rob_entry_executed((*aentry)->matrix_sbe.get_rob_entry());
                    matrix_engine->matrix_rob->set_md_entry_valid((*aentry)->matrix_sbe.md, (*aentry)->matrix_sbe.dst_memdep_idx);
                    // auto it = Arithmetic_Queue.begin() + i;
                    auto aentry_to_delete = *aentry;
                    aentry = Arithmetic_Queue.erase(aentry);
                    if(aentry_to_delete != nullptr) {
                        delete aentry_to_delete;
                    }
                //====等会改，改完了现在

                    // delete aentry->matrix_sbe;
                    // delete aentry;
            }else {
                //Have not been issued
                bool ready_to_issue = false;
                if((*aentry)->matrix_sbe._minst->isMatrixInstArith()){
                    // ready_to_issue = matrix_engine->matrix_rename->get_RAT_vld(aentry->matrix_sbe.get_renamed_src1()) && matrix_engine->matrix_rename->get_RAT_vld(aentry->matrix_sbe.get_renamed_src2());
                    ready_to_issue = matrix_engine->matrix_rename->get_PR_vld((*aentry)->matrix_sbe.get_renamed_src1()) && matrix_engine->matrix_rob->raw_solved((*aentry)->matrix_sbe.ms1, (*aentry)->matrix_sbe.src1_memdep_idx);
                    ready_to_issue = ready_to_issue && matrix_engine->matrix_rename->get_PR_vld((*aentry)->matrix_sbe.get_renamed_src2()) && matrix_engine->matrix_rob->raw_solved((*aentry)->matrix_sbe.ms2, (*aentry)->matrix_sbe.src2_memdep_idx);
                    //为了让第一次访问acc的时候能够正常使用
                    if((*aentry)->matrix_sbe._minst->ismmacc()){
                        if (matrix_engine->matrix_rename->accInitial[(*aentry)->matrix_sbe.get_dst_prf_num()-tileReg_num] == true){
                            ready_to_issue = ready_to_issue && true;
                            if (ready_to_issue){
                                matrix_engine->matrix_rename->accInitial[(*aentry)->matrix_sbe.get_dst_prf_num()-tileReg_num] = false;
                            }
                        } else {
                            ready_to_issue = ready_to_issue;
                        }
                        ready_to_issue = ready_to_issue && (matrix_engine->matrix_rename->check_readLock((*aentry)->matrix_sbe.get_dst_prf_num() , (*aentry)->matrix_sbe.src3_read_idx));
                    }
                    else if(((*aentry)->matrix_sbe.get_renamed_src1() == (*aentry)->matrix_sbe.get_dst_prf_num()) || ((*aentry)->matrix_sbe.get_renamed_src2() == (*aentry)->matrix_sbe.get_dst_prf_num())) {
                        ready_to_issue = ready_to_issue && (matrix_engine->matrix_rename->check_readLock((*aentry)->matrix_sbe.get_dst_prf_num(), (*aentry)->matrix_sbe.dst_read_idx));
                    }
                    // DPRINTF(MatrixDispatcher, "This aentry need check %d is %d, %d is %d \n", (*aentry)->matrix_sbe.get_renamed_src1(), matrix_engine->matrix_rename->get_PR_vld((*aentry)->matrix_sbe.get_renamed_src1()), (*aentry)->matrix_sbe.get_renamed_src2(), matrix_engine->matrix_rename->get_PR_vld((*aentry)->matrix_sbe.get_renamed_src2()));
                    // DPRINTF(MatrixDispatcher, "ready to issue is %d\n", ready_to_issue);
                }
                else if((*aentry)->matrix_sbe._minst->isMatrixRedc() || (*aentry)->matrix_sbe._minst->ismlut()) {
                    if((*aentry)->matrix_sbe.get_renamed_src1() == (*aentry)->matrix_sbe.get_dst_prf_num())
                        ready_to_issue = matrix_engine->matrix_rename->check_readLock((*aentry)->matrix_sbe.get_dst_prf_num(), (*aentry)->matrix_sbe.dst_read_idx);
                    ready_to_issue = ready_to_issue && matrix_engine->matrix_rename->get_PR_vld((*aentry)->matrix_sbe.get_renamed_src1()) && matrix_engine->matrix_rob->raw_solved((*aentry)->matrix_sbe.ms1, (*aentry)->matrix_sbe.src1_memdep_idx);
                    // DPRINTF(MatrixDispatcher, "ready to issue is %d\n", ready_to_issue);
                }
                //May be there will be more types of instructions here
                if(ready_to_issue || (*aentry)->matrix_sbe.canIssue){
                    if(!(*aentry)->matrix_sbe._minst->ismmacc())
                        matrix_engine->matrix_rename->set_PR_vld((*aentry)->matrix_sbe.get_dst_prf_num(), false);
                    (*aentry)->matrix_sbe.canIssue = true;
                    bool success = false;
                    if((*aentry)->matrix_sbe._minst->ismmacc()){
                        uint8_t lane; // used to choose one lane to issue the inst!
                        for(uint8_t j = 0; j < matrix_engine->lane_num; j++){
                            if(matrix_engine->matrix_lanes[j]->isOccupied() == false){
                                success = true;
                                lane = j;
                                break;
                            }
                        }
                        if(success){
                            (*aentry)->matrix_sbe.isIssue = true;
                            (*aentry)->matrix_sbe.lane_idx = lane;
                            (*aentry)->matrix_sbe.ewu_idx = lane;
                            matrix_engine->matrix_lanes[lane]->issue_inst(&((*aentry)->matrix_sbe));
                            matrix_engine->matrix_lanes[lane]->set_matrixrename(matrix_engine->matrix_rename);
                            // DPRINTF(MatrixDispatcher, "Send instruction to the Lane:%d\n", lane);
                            DPRINTF(MatrixDispatcherReadLock, "AQ: issue mmacc inst, readrls src1 pr idx %d, read idx = %d\n", (*aentry)->matrix_sbe.get_dst_prf_num(), (*aentry)->matrix_sbe.md);
                            DPRINTF(MatrixDispatcher, "send inst to lane %d, md = %d, src3_memdep_idx = %d, phy dst = %d\n", lane, (*aentry)->matrix_sbe.md, (*aentry)->matrix_sbe.dst_memdep_idx, (*aentry)->matrix_sbe.get_dst_prf_num());
                            AQread++;
                        } else{
                            waitforlane = true;
                        }
                    } else if ((*aentry)->matrix_sbe._minst->ismadd()||(*aentry)->matrix_sbe._minst->ismsub()||(*aentry)->matrix_sbe._minst->ismmul()||(*aentry)->matrix_sbe._minst->ismmulh()||(*aentry)->matrix_sbe._minst->ismmax()||(*aentry)->matrix_sbe._minst->ismumax()||(*aentry)->matrix_sbe._minst->ismmin()||(*aentry)->matrix_sbe._minst->ismumin()||(*aentry)->matrix_sbe._minst->ismsll()||(*aentry)->matrix_sbe._minst->ismsrl()||(*aentry)->matrix_sbe._minst->ismsra()) {
                        uint8_t ewu_idx; // used to choose one lane to issue the inst!
                        for(uint8_t j = 0; j < matrix_engine->lane_num; j++){
                            if(matrix_engine->ew_unit->isOccupied(j) == false){
                                success = true;
                                ewu_idx = j;
                                break;
                            }
                        }
                        if(success){
                            if(matrix_engine->ew_unit->isIdle() == true){
                                matrix_engine->ew_unit->startTicking();
                            }
                            (*aentry)->matrix_sbe.isIssue = true;
                            (*aentry)->matrix_sbe.ewu_idx = ewu_idx;
                            (*aentry)->matrix_sbe.lane_idx = ewu_idx;
                            matrix_engine->ew_unit->recv_opera((*aentry)->matrix_sbe, ewu_idx, (*aentry)->matrix_sbe._minst->isSigned());
                            DPRINTF(MatrixDispatcherReadLock, "AQ: issue ew inst, readrls dst pr idx %d, lpr = %d\n", (*aentry)->matrix_sbe.get_dst_prf_num(), (*aentry)->matrix_sbe.md);
                        } else{
                            waitforewu = true;
                        }
                    } else if((*aentry)->matrix_sbe._minst->ismredcadd() || (*aentry)->matrix_sbe._minst->ismredcmax()) {
                        uint8_t ewu_idx; // used to choose one lane to issue the inst!
                        for(uint8_t j = 0; j < matrix_engine->lane_num; j++){
                            if(matrix_engine->ew_unit->isOccupied(j) == false){
                                success = true;
                                ewu_idx = j;
                                break;
                            }
                        }
                        if(success){
                            if(matrix_engine->ew_unit->isIdle() == true){
                                matrix_engine->ew_unit->startTicking();
                            }
                            (*aentry)->matrix_sbe.isIssue = true;
                            (*aentry)->matrix_sbe.ewu_idx = ewu_idx;
                            (*aentry)->matrix_sbe.lane_idx = ewu_idx;
                            matrix_engine->ew_unit->redc_req((*aentry)->matrix_sbe, ewu_idx, (*aentry)->matrix_sbe._minst->isSigned());
                            DPRINTF(MatrixDispatcherReadLock, "AQ: issue redc inst, readrls dst pr idx %d, lpr = %d\n", (*aentry)->matrix_sbe.get_dst_prf_num(), (*aentry)->matrix_sbe.md);
                        } else{
                            waitforewu = true;
                        }
                    } else if((*aentry)->matrix_sbe._minst->ismlut()) {
                        uint32_t* matrix_src = new uint32_t[8];
                        uint32_t* matrix_dst = new uint32_t[8];
                        for(uint32_t j = 0; j < (*aentry)->matrix_sbe.get_cfg_sizeN(); j++){
                            matrix_src[j] = matrix_engine->matrix_reg->rdreg_int32((*aentry)->matrix_sbe.get_renamed_src1(), j / 2, (*aentry)->matrix_sbe._minst->uimm3(), j % 2);
                        }
                        if(matrix_engine->matrix_spm->readLut(matrix_src, matrix_dst, (*aentry)->matrix_sbe.get_cfg_sizeN(), (*aentry)->matrix_sbe._minst->ms2())) {
                            (*aentry)->matrix_sbe.isDone = true;
                            for(uint32_t j = 0; j < (*aentry)->matrix_sbe.get_cfg_sizeN(); j++){
                                this->matrix_engine->matrix_reg->wtreg_int32((*aentry)->matrix_sbe.get_dst_prf_num(), j / 2, (*aentry)->matrix_sbe._minst->uimm3(), j % 2, matrix_dst[j]);
                            }
                        }
                        delete[] matrix_src;
                        delete[] matrix_dst;
                    }
                } else {
                    for(uint8_t j = 0; j < matrix_engine->lane_num; j++){
                        if(matrix_engine->matrix_lanes[j]->isOccupied() == false){
                            freelane_waitinst = true;
                            break;
                        }
                    }
                }
                ++aentry;
            }

        }
        if(freelane_waitinst){
            LaneWaitingForReg++;
        }
        if(waitforlane){
            WaitingForLane++;
        }
        if(waitforewu){
            WaitingForEWU++;
        }
    } else {
        // DPRINTF(MatrixDispatcher, "AQ size = %d\n", Arithmetic_Queue.size());
    }


    // //要用rename reorder以及mmu和matrix lane
    // if(issueMemReq){
    //     MemQueueEntry* mentry = Memory_Queue.front();
    //     matrix_rename
    //     matrix_reorder_buffer
    //     if(mentry->minst.isLoad()){
            
    //         matrix_mmu->readMatrixMem()
    //     } else if(mentry->minst.isStore()){
    //         matrix_mmu->writeMatrixMem()
    //     }
    //     Memory_Queue.pop_front();
    //     delete mentry;
    // } else if(issueArithReq){

    // }


    //start dispatch rename reorder

    //update status


}
}//namespace gem5

