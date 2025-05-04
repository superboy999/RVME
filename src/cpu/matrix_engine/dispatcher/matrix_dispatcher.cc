// ==========================================
// Author: cwq
// Last Date: 2024/2/28
// Description: Decode, and Will send to
// queue first, reorder and rename! Finally 
// send to functional unit!
// ==========================================

#include "cpu/matrix_engine/dispatcher/matrix_dispatcher.hh"
#include "debug/MatrixDispatcher.hh"

#include <cassert>
#include <cstdint>
#include <string>

namespace gem5
{

MatrixDispatcher::MatrixDispatcher(const MatrixDispatcherParams &params) :
TickedObject(params), MQ_depth(params.MQ_depth), AQ_depth(params.AQ_depth), 
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
    allLaneisFree
    .name(name() + ".TicksForAllLaneFree")
    .desc("Ticks of all Matrix Lane is free");
    LaneisFree
    .name(name() + ".TicksForLaneFree")
    .desc("Ticks of Matrix Lane is free");
    LaneWaitingForReg
    .name(name() + ".TicksForLaneWaitingReg")
    .desc("Ticks of Lane waiting for reg");
    hasFreeReg
    .name(name() + ".TicksForFreeReg")
    .desc("Ticks of free reg");
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
    testnum
    .name(name() + ".testnum")
    .desc("testnum");   
}

void MatrixDispatcher::renameMatrixInst(RiscvISA::RiscvMatrixInst &minst, ScoreBoard_Entry* matrix_sbe)
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
    uint16_t load_md = minst.load_md();
    matrix_sbe->set_MatrixStaticInst(&minst);

    // matrix_engine->matrix_rename->set_RAT_vld(load_md, false); //FIXED: every md before inst finished, will be marked as invalid right behind selected!

    if(minst.isLoad()||minst.isStore()){
        if(minst.isLoad()){
            //This function has destreg, so rename here.
            // mld<b/h/w/d> md, rs2, (rs1)
            pmd = matrix_engine->matrix_rename->get_freeReg();
            matrix_engine->matrix_rename->regLock(pmd);
            old_dst = matrix_engine->matrix_rename->get_preg_RAT(load_md);
            matrix_engine->matrix_rename->set_preg_RAT(load_md, pmd);
            matrix_sbe->set_dst_lrf_num(load_md);
            matrix_sbe->set_dst_prf_num(pmd);
            matrix_sbe->set_old_dst(old_dst);
            matrix_load_inst++;
        } else if(minst.isStore()){
            //This function has no destination register, so dont rename;
            //mst<b/h/w/d> ms3, rs2, (rs1)
            pms3 = matrix_engine->matrix_rename->get_preg_RAT(ms3);
            matrix_engine->matrix_rename->regLock(pms3);
            matrix_sbe->set_renamed_src3(pms3);
            matrix_store_inst++;
        }
        matrix_sbe->set_cfg_size(matrix_engine->get_sizeN(), matrix_engine->get_sizeM(), matrix_engine->get_sizeK());
    } else if(minst.isMatrixInstArith()){
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
        pmd = matrix_engine->matrix_rename->get_freeReg();
        matrix_engine->matrix_rename->regLock(pmd);
        old_dst = matrix_engine->matrix_rename->get_preg_RAT(md);
        matrix_engine->matrix_rename->set_preg_RAT(md, pmd);
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
    }
}

void MatrixDispatcher::set_rob_entry(RiscvISA::RiscvMatrixInst &minst, ScoreBoard_Entry* matrix_sbe)
{
    assert(!matrix_engine->matrix_rob->rob_full());
    if(!matrix_engine->matrix_rob->isOccupied()){
        matrix_engine->matrix_rob->startTicking();
    }
    bool old_dst_vld = false;
    if(minst.isStore()){
        old_dst_vld = false;
    } else{
        old_dst_vld = true;
    }
    if(!minst.isStore()){
        uint32_t rob_entry = matrix_engine->matrix_rob->set_rob_entry(matrix_sbe->get_dst_lrf_num(), matrix_sbe->get_dst_prf_num(), matrix_sbe->get_old_dst(), old_dst_vld);
        matrix_sbe->set_rob_entry(rob_entry);
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
            matrix_engine->cfgSizeK(matrix_sbe->_minst->uimm7()); 
        }
        if(matrix_sbe->_minst->ismcfgmi()){
            matrix_engine->cfgSizeM(matrix_sbe->_minst->uimm7());
        }
        if(matrix_sbe->_minst->ismcfgni()){
            matrix_engine->cfgSizeN(matrix_sbe->_minst->uimm7());
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
    if(minst.isLoad()||minst.isStore())
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
    bool free_prf_avaliable = !matrix_engine->matrix_rename->freeList_empty();
    DPRINTF(MatrixDispatcher, "rob:%d, M/AQ:%d, free_pr:%d\n", rob_entry_notfull, queue_notfull, free_prf_avaliable);
    if(!free_prf_avaliable){
        WaitingForReg++;
    }
    return rob_entry_notfull&&queue_notfull&&free_prf_avaliable; //Delete config stall
    // return rob_entry_notfull&&queue_notfull&&free_prf_avaliable&&!config_stall;//FIXME: check config_stall correctness.
}

bool MatrixDispatcher::dispatchGrant_withoutReg(RiscvISA::RiscvMatrixInst &minst)
{
    bool rob_entry_notfull = !matrix_engine->matrix_rob->rob_full();
    bool queue_notfull = false;
    if(minst.isLoad()||minst.isStore())
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
    DPRINTF(MatrixDispatcher, "rob:%d, M/AQ:%d\n", rob_entry_notfull, queue_notfull);

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
}

void MatrixDispatcher::stopTicking()
{
    stop();
    isBusy = false;
}

bool MatrixDispatcher::isOccupied()
{
    return isBusy;
}
void MatrixDispatcher::evaluate()
{
    bool free_prf_avaliable = !matrix_engine->matrix_rename->freeList_empty();

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

    if(!matrix_engine->matrix_rename->freeList_empty()){
        hasFreeReg++;
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
    //         matrix_engine->cfgSizeK(matrix_sbe->_minst->uimm7()); 
    //     }
    //     if(matrix_sbe->_minst->ismcfgmi()){
    //         matrix_engine->cfgSizeM(matrix_sbe->_minst->uimm7());
    //     }
    //     if(matrix_sbe->_minst->ismcfgni()){
    //         matrix_engine->cfgSizeN(matrix_sbe->_minst->uimm7());
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
        ready_to_issue = matrix_engine->matrix_rename->get_PR_vld((*mentry)->matrix_sbe.get_renamed_src3());
        MQread++;
    }
    if(ready_to_issue || (*mentry)->matrix_sbe.isSent){
        DPRINTF(MatrixDispatcher, "memory queue idx %u is sent!\n", index);
        (*mentry)->matrix_sbe.isSent = true;
        bool success = false;
        uint64_t addr_index;
        addr_index = (*mentry)->matrix_sbe.get_rs1_value() + ((*mentry)->matrix_sbe.get_rs2_value())*((*mentry)->matrix_sbe.get_executed_num());
        if((*mentry)->matrix_sbe._minst->isLoad() && !(*mentry)->matrix_sbe.isSendAndWait){
            if(XY){
                // // occupy write port
                // if(!matrix_engine->matrix_reg->occupy_wtport()){
                //     ++mentry;
                //     continue;
                // }
                //Load MB once a row!
                success = matrix_engine->matrix_mmu->readMatrixMem(addr_index, (*mentry)->matrix_sbe.get_rs2_value(), &((*mentry)->tc), 0, (*mentry)->matrix_sbe.get_dst_prf_num(), [this, matrix_sbe_copy = (*(*mentry)).matrix_sbe, mentry_copy = (*mentry)](uint8_t *data, uint8_t size){
                    for(uint8_t i = 0; i < matrix_sbe_copy.get_rs2_value(); i++){
                        this->matrix_engine->matrix_reg->wtreg_byte(matrix_sbe_copy.get_dst_prf_num(), i % (this->matrix_engine->matrix_reg->bank_num), i / (this->matrix_engine->matrix_reg->bank_num), matrix_sbe_copy.get_executed_num(), data[i]);
                    }
                    this->matrix_engine->matrix_reg->rls(matrix_sbe_copy.get_dst_prf_num());
                    if(data[31] == 0){
                        DPRINTF(MatrixDispatcher, "find zero in loading stage!!\n");
                    }
                    mentry_copy->matrix_sbe.load();
                });
            } else {
                // // occupy write port
                // if(!matrix_engine->matrix_reg->occupy_wtport()){
                //     ++mentry;
                //     continue;
                // }
                //Load MA once a row!
                success = matrix_engine->matrix_mmu->readMatrixMem(addr_index, (*mentry)->matrix_sbe.get_rs2_value(), &((*mentry)->tc), 0, (*mentry)->matrix_sbe.get_dst_prf_num(), [this, matrix_sbe_copy = (*(*mentry)).matrix_sbe, mentry_copy = (*mentry)](uint8_t *data, uint8_t size){
                    for(uint8_t i = 0; i < matrix_sbe_copy.get_rs2_value(); i++){
                        this->matrix_engine->matrix_reg->wtreg_byte(matrix_sbe_copy.get_dst_prf_num(), i % (this->matrix_engine->matrix_reg->bank_num), i / (this->matrix_engine->matrix_reg->bank_num), matrix_sbe_copy.get_executed_num(), data[i]);
                    }
                    this->matrix_engine->matrix_reg->rls(matrix_sbe_copy.get_dst_prf_num());
                    mentry_copy->matrix_sbe.load();
                });
            }
            if(success){
                (*mentry)->matrix_sbe.execute();
                success = false; //reset success signal
                matrix_engine->matrix_rename->set_PR_vld((*mentry)->matrix_sbe.get_dst_prf_num(), false);
                // DPRINTF(MatrixDispatcher, "pr idx %d is valid now\n", (*mentry)->matrix_sbe.get_dst_prf_num());
                if(XY){
                    if ((*mentry)->matrix_sbe.get_executed_num() == (*mentry)->matrix_sbe.get_cfg_sizeN()){
                        (*mentry)->matrix_sbe.isSendAndWait = true;
                    }
                } else {
                    if ((*mentry)->matrix_sbe.get_executed_num() == (*mentry)->matrix_sbe.get_cfg_sizeM()){
                        (*mentry)->matrix_sbe.isSendAndWait = true;
                    }
                }
                break;
            }
        } else if((*mentry)->matrix_sbe._minst->isStore() && !(*mentry)->matrix_sbe.isSendAndWait){
            if(XY){
                //Store MB once a row!
                if(!matrix_engine->matrix_reg->try_occupy((*mentry)->matrix_sbe.get_renamed_src3())){
                    ++mentry;
                    continue;
                }
                uint8_t* mreg_row_data = new uint8_t[(*mentry)->matrix_sbe.get_rs2_value()];
                for(uint32_t j = 0; j < (*mentry)->matrix_sbe.get_rs2_value(); j++){
                    mreg_row_data[j] = matrix_engine->matrix_reg->rdreg_byte((*mentry)->matrix_sbe.get_renamed_src3(), j % matrix_engine->matrix_reg->bank_num, j / matrix_engine->matrix_reg->bank_num, (*mentry)->matrix_sbe.get_executed_num());
                }
                success = matrix_engine->matrix_mmu->writeMatrixMem(addr_index, mreg_row_data, (*mentry)->matrix_sbe.get_rs2_value(), &((*mentry)->tc), 0, [this, matrix_sbe_copy = (*(*mentry)).matrix_sbe, mentry_copy = (*mentry), mreg_row_data](){
                    this->matrix_engine->matrix_reg->rls(matrix_sbe_copy.get_renamed_src3());
                    mentry_copy->matrix_sbe.store();
                    delete[] mreg_row_data; // move this here to ensure it is deleted after store operation
                });
            } else {
                //Store MA once a row!
                if(!matrix_engine->matrix_reg->try_occupy((*mentry)->matrix_sbe.get_renamed_src3())){
                    ++mentry;
                    continue;
                }
                uint8_t* mreg_row_data = new uint8_t[(*mentry)->matrix_sbe.get_rs2_value()];
                for(uint32_t j = 0; j < (*mentry)->matrix_sbe.get_rs2_value(); j++){
                    mreg_row_data[j] = matrix_engine->matrix_reg->rdreg_byte((*mentry)->matrix_sbe.get_renamed_src3(), j % matrix_engine->matrix_reg->bank_num, j / matrix_engine->matrix_reg->bank_num, (*mentry)->matrix_sbe.get_executed_num());
                }
                success = matrix_engine->matrix_mmu->writeMatrixMem(addr_index, mreg_row_data, (*mentry)->matrix_sbe.get_rs2_value(), &((*mentry)->tc), 0, [this, matrix_sbe_copy = (*(*mentry)).matrix_sbe, mentry_copy = (*mentry), mreg_row_data](){
                    this->matrix_engine->matrix_reg->rls(matrix_sbe_copy.get_renamed_src3());
                    mentry_copy->matrix_sbe.store();
                    delete[] mreg_row_data; // move this here to ensure it is deleted after store operation
                });
            }
            if(success){
                (*mentry)->matrix_sbe.execute();
                success = false; // reset success signal

                if(XY){
                    if ((*mentry)->matrix_sbe.get_executed_num() == (*mentry)->matrix_sbe.get_cfg_sizeM()){
                        (*mentry)->matrix_sbe.isSendAndWait = true;
                    }
                } else {
                    if ((*mentry)->matrix_sbe.get_executed_num() == (*mentry)->matrix_sbe.get_cfg_sizeM()){
                        (*mentry)->matrix_sbe.isSendAndWait = true;
                    }
                }
                break;
            }
        }

        if(XY){
            if((*mentry)->matrix_sbe.isSendAndWait && (*mentry)->matrix_sbe.get_load_cnt() == (*mentry)->matrix_sbe.get_cfg_sizeN()){
                // load commit
                matrix_engine->matrix_rename->set_PR_vld((*mentry)->matrix_sbe.get_dst_prf_num(), true);
                matrix_engine->matrix_rename->regrls((*mentry)->matrix_sbe.get_dst_prf_num());
                // matrix_engine->matrix_reg->rls((*mentry)->matrix_sbe.get_dst_prf_num());
                // matrix_engine->matrix_reg->rls_wrport();
                DPRINTF(MatrixDispatcher, "MQ: pr idx %d is valid now\n", (*mentry)->matrix_sbe.get_dst_prf_num());
                matrix_engine->matrix_rob->set_rob_entry_executed((*mentry)->matrix_sbe.get_rob_entry());
                mentry = Memory_Queue.erase(mentry);
                XY = false;//try to not depend on the config to reset the XY
            } else if((*mentry)->matrix_sbe.isSendAndWait && (*mentry)->matrix_sbe.get_store_cnt() == (*mentry)->matrix_sbe.get_cfg_sizeM()) {
                // store commit
                // matrix_engine->matrix_reg->rls((*mentry)->matrix_sbe.get_dst_prf_num());
                // matrix_engine->matrix_rob->set_rob_entry_executed((*mentry)->matrix_sbe.get_rob_entry());
                matrix_engine->matrix_rename->regrls((*mentry)->matrix_sbe.get_renamed_src3());
                std::cout << "Before erase, size: " << Memory_Queue.size() << std::endl;
                mentry = Memory_Queue.erase(mentry);
                std::cout << "After erase, size: " << Memory_Queue.size() << std::endl;
            } else {
                ++mentry;
            }
        } else {
            //load commit
            if((*mentry)->matrix_sbe.isSendAndWait && (*mentry)->matrix_sbe.get_load_cnt() == (*mentry)->matrix_sbe.get_cfg_sizeM()){
                matrix_engine->matrix_rename->set_PR_vld((*mentry)->matrix_sbe.get_dst_prf_num(), true);
                matrix_engine->matrix_rename->regrls((*mentry)->matrix_sbe.get_dst_prf_num());
                // matrix_engine->matrix_reg->rls_wrport();
                // matrix_engine->matrix_reg->rls((*mentry)->matrix_sbe.get_dst_prf_num());
                DPRINTF(MatrixDispatcher, "MQ: pr idx %d is valid now\n", (*mentry)->matrix_sbe.get_dst_prf_num());
                matrix_engine->matrix_rob->set_rob_entry_executed((*mentry)->matrix_sbe.get_rob_entry());
                mentry = Memory_Queue.erase(mentry);
                XY = true;
            } else if((*mentry)->matrix_sbe.isSendAndWait && (*mentry)->matrix_sbe.get_store_cnt() == (*mentry)->matrix_sbe.get_cfg_sizeM()){
                //store commit 
                // matrix_engine->matrix_reg->rls((*mentry)->matrix_sbe.get_dst_prf_num());
                // matrix_engine->matrix_rob->set_rob_entry_executed((*mentry)->matrix_sbe.get_rob_entry());
                matrix_engine->matrix_rename->regrls((*mentry)->matrix_sbe.get_renamed_src3());
                std::cout << "Before erase, size: " << Memory_Queue.size() << std::endl;
                mentry = Memory_Queue.erase(mentry);
                std::cout << "After erase, size: " << Memory_Queue.size() << std::endl;
            } else{
                ++mentry;
            }
        }
    } else {
        ++mentry;
    }
}

    }

    if(Arithmetic_Queue.size()!=0)
    {
        // ArithQueueEntry* aentry;
        // uint32_t issue_range = (OoO) ? AQ_depth : 1;
        // uint32_t issue_range = (OoO) ? Arithmetic_Queue.size() : 1;
        DPRINTF(MatrixDispatcher, "check aq!!\n");
        // for(uint32_t i = 0; i < issue_range; i++)
        bool freelane_waitinst = false;
        bool waitforlane = false;
        for(auto aentry = Arithmetic_Queue.begin(); aentry != Arithmetic_Queue.end();)
        {
            // aentry = Arithmetic_Queue[i];

            //judge the inst status
            if((*aentry)->matrix_sbe.isIssue&&(!(*aentry)->matrix_sbe.isDone)){
                //just issued but have not done
                if(matrix_engine->matrix_lanes[(*aentry)->matrix_sbe.lane_idx]->tranz_minst.instDone){
                    (*aentry)->matrix_sbe.isDone = true;
                }
                ++aentry;
            }else if((*aentry)->matrix_sbe.isDone){
                    // inst has been executed
                    matrix_engine->matrix_rename->set_PR_vld((*aentry)->matrix_sbe.get_dst_prf_num(), true);
                    matrix_engine->matrix_rename->regrls((*aentry)->matrix_sbe.get_dst_prf_num());
                    matrix_engine->matrix_rename->regrls((*aentry)->matrix_sbe.get_renamed_src1());
                    matrix_engine->matrix_rename->regrls((*aentry)->matrix_sbe.get_renamed_src2());
                    DPRINTF(MatrixDispatcher, "AQ: pr idx %d is valid now\n", (*aentry)->matrix_sbe.get_dst_prf_num());
                    matrix_engine->matrix_rob->set_rob_entry_executed((*aentry)->matrix_sbe.get_rob_entry());
                    // auto it = Arithmetic_Queue.begin() + i;
                    aentry = Arithmetic_Queue.erase(aentry);


                    // delete aentry->matrix_sbe;
                    // delete aentry;
            }else {
                //Have not been issued
                bool ready_to_issue = false;
                if((*aentry)->matrix_sbe._minst->isMatrixInstArith()){
                    // ready_to_issue = matrix_engine->matrix_rename->get_RAT_vld(aentry->matrix_sbe.get_renamed_src1()) && matrix_engine->matrix_rename->get_RAT_vld(aentry->matrix_sbe.get_renamed_src2());
                    ready_to_issue = matrix_engine->matrix_rename->get_PR_vld((*aentry)->matrix_sbe.get_renamed_src1()) && matrix_engine->matrix_rename->get_PR_vld((*aentry)->matrix_sbe.get_renamed_src2());

                    DPRINTF(MatrixDispatcher, "This aentry need check %d is %d, %d is %d \n", (*aentry)->matrix_sbe.get_renamed_src1(), matrix_engine->matrix_rename->get_PR_vld((*aentry)->matrix_sbe.get_renamed_src1()), (*aentry)->matrix_sbe.get_renamed_src2(), matrix_engine->matrix_rename->get_PR_vld((*aentry)->matrix_sbe.get_renamed_src2()));
                    DPRINTF(MatrixDispatcher, "ready to issue is %d\n", ready_to_issue);
                }
                //May be there will be more types of instructions here
                if(ready_to_issue){
                    bool success = false;
                    uint8_t lane; // used to choose one lane to issue the inst!
                    if((*aentry)->matrix_sbe._minst->isMatrixInstArith()){
                        for(uint8_t j = 0; j < matrix_engine->lane_num; j++){
                            if(matrix_engine->matrix_lanes[j]->isOccupied() == false){
                                success = true;
                                lane = j;
                                break;
                            }
                        }
                        if(success){
                            matrix_engine->matrix_lanes[lane]->issue_inst(&((*aentry)->matrix_sbe));
                            (*aentry)->matrix_sbe.isIssue = true;
                            (*aentry)->matrix_sbe.lane_idx = lane;
                            DPRINTF(MatrixDispatcher, "Send instruction to the Lane:%d\n", lane);
                            AQread++;
                        } else{
                            waitforlane = true;
                        }
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

