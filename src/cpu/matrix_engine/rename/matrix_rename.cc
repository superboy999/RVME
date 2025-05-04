/*
 * @Author: superboy
 * @Date: 2024-05-30 01:21:06
 * @LastEditTime: 2025-03-26 01:25:01
 * @LastEditors: superboy
 * @Description: 
 * @FilePath: /gem5-rvm/src/cpu/matrix_engine/rename/matrix_rename.cc
 * 
 */
#include "cpu/matrix_engine/rename/matrix_rename.hh"
#include "debug/MatrixRename.hh"

#include <cstdint>
#include <cassert>
#include <iostream>

namespace gem5
{

MatrixRename::MatrixRename(const MatrixRenameParams &params):
    SimObject(params), numPhysicalRegs(params.numPhysicalRegs),
    numLogicalRegs(params.numLogicalRegs)
{
    DPRINTF(MatrixRename, "Created the Renaming Unit object!\n");
    for(uint32_t i = 0; i < numPhysicalRegs; i++){
        Matrix_FreeList.push_back(i);
    }//default the physical register from 0-7 has been alias to the logical register!
    PR_vld.resize(numPhysicalRegs, false);
    PR_using.resize(numPhysicalRegs, 0);
    for(uint32_t i = 0; i < numLogicalRegs; i++){
        RegAliasTable[i] = 99; //fix bug
        // RAT_vld[i] = true;
    }
    for (uint8_t i = 0; i < numPhysicalRegs; i++){
        PR_vld[i] = false; // fix the memory queue OoO
        PR_using[i] = 0;
    }
}

MatrixRename::~MatrixRename()
{}

void MatrixRename::regStats()
{
    SimObject::regStats();
    RegUse
    .name(name() + ".numRegisterUse")
    .init(numPhysicalRegs)
    .desc("Count the register use number");
    RAT_read
    .name(name() + ".numRATRead")
    .desc("Count the number of RAT read");
    RAT_write
    .name(name() + ".numRATWrite")
    .desc("Count the number of RAT write");
    FreeList_read
    .name(name() + ".numFreeListRead")
    .desc("Count the number of FreeList read");
    FreeList_write
    .name(name() + ".numFreeListWrite")
    .desc("Count the number of FreeList write");
}

bool MatrixRename::freeList_empty()
{
    if(Matrix_FreeList.size()==0){
        return true;
    } else{
        return false;
    }
}

uint32_t MatrixRename::freeList_size()
{
    return Matrix_FreeList.size();
}

uint32_t MatrixRename::get_freeReg()
{
    assert(!(Matrix_FreeList.size()==0));
    uint32_t freeReg;
    freeReg = Matrix_FreeList.front();
    std::cout << "Size before pop: " << Matrix_FreeList.size() << std::endl;
    Matrix_FreeList.pop_front();
    std::cout << "Size before pop: " << Matrix_FreeList.size() << std::endl;
    RegUse[freeReg]++;
    FreeList_read++;
    return freeReg;
}

//均是防止随意free register
void MatrixRename::regLock(uint32_t phyreg_idx)
{
    PR_using[phyreg_idx]++;
    std::cout << "PR_use" << phyreg_idx << ": "<< PR_using[phyreg_idx] << std::endl;
}

void MatrixRename::regrls(uint32_t phyreg_idx)
{
    assert(PR_using[phyreg_idx]>0);
    PR_using[phyreg_idx]--;
    std::cout << "PR_rls" << phyreg_idx << ": "<< PR_using[phyreg_idx] << std::endl;
}

bool MatrixRename::checkLock(uint32_t phyreg_idx)
{
    return (PR_using[phyreg_idx]>0);
}


void MatrixRename::set_freeReg(uint32_t phyreg_idx)
{
    FreeList_write++;
    Matrix_FreeList.push_back(phyreg_idx);
}

uint32_t MatrixRename::get_preg_RAT(uint32_t logreg_idx)
{
    RAT_read++;
    return RegAliasTable[logreg_idx];
}

void MatrixRename::set_preg_RAT(uint32_t logreg_idx, uint32_t phyreg_idx)
{
    RegAliasTable[logreg_idx] = phyreg_idx;
    set_PR_vld(phyreg_idx, false);
    RAT_write++;
    DPRINTF(MatrixRename, "Set PR:%d to false\n", phyreg_idx);
}

bool MatrixRename::get_PR_vld(uint8_t phyreg_idx)
{
    return PR_vld[phyreg_idx];
}

void MatrixRename::set_PR_vld(uint8_t phyreg_idx, bool vld)
{
    PR_vld[phyreg_idx] = vld;
    DPRINTF(MatrixRename, "Set PR:%d to %d\n", phyreg_idx, vld);
}

void MatrixRename::print_RAT()
{
    DPRINTF(MatrixRename, "REGISTER ALIAS TABLE\n");
    DPRINTF(MatrixRename, "%u %u %u %u %u %u %u %u\n"
        ,RegAliasTable[0],RegAliasTable[1],RegAliasTable[2],RegAliasTable[3],
        RegAliasTable[4],RegAliasTable[5],RegAliasTable[6],RegAliasTable[7]);
}

void MatrixRename::print_pr_vld()
{
    DPRINTF(MatrixRename, "PHYSICAL REGISTER VALID\n");
    DPRINTF(MatrixRename, "%u %u %u %u %u %u %u %u %u %u %u %u %u %u %u %u\n"
        ,PR_vld[0],PR_vld[1],PR_vld[2],PR_vld[3],
        PR_vld[4],PR_vld[5],PR_vld[6],PR_vld[7],
        PR_vld[8], PR_vld[9], PR_vld[10], PR_vld[11],
        PR_vld[12], PR_vld[13], PR_vld[14], PR_vld[15]);
}

} // namespace gem5