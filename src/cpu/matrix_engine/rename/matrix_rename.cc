/*
 * @Author: superboy
 * @Date: 2024-05-30 01:21:06
 * @LastEditTime: 2025-10-18 17:00:29
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
    numLogicalRegs(params.numLogicalRegs), tilePRFs(params.tilePRFs),
    accPRFs(params.accPRFs)
{
    DPRINTF(MatrixRename, "Created the Renaming Unit object!\n");
    for(uint32_t i = 0; i < tilePRFs; i++){
        Matrix_FreeTileList.push_back(i);
    }//default the physical register from 0-7 has been alias to the logical register!
    for(uint32_t i = tilePRFs+4; i < numPhysicalRegs; i++){
        Matrix_FreeAccList.push_back(i);
    }
    PR_vld.resize(numPhysicalRegs, false);
    PR_using.resize(numPhysicalRegs, 0);
    for(uint32_t i = 0; i < numLogicalRegs-4; i++){
        RegAliasTable[i] = 99; //fix bug
        // RAT_vld[i] = true;
    }
    RegAliasTable[4] = 16;
    RegAliasTable[5] = 17;
    RegAliasTable[6] = 18;
    RegAliasTable[7] = 19;
    for (uint8_t i = 0; i < numPhysicalRegs; i++){
        PR_vld[i] = false; // fix the memory queue OoO
        PR_using[i] = 0;
    }
    accRlsEn.resize(4, false);
    accInitial.resize(4, true);
}

MatrixRename::~MatrixRename()
{}

bool MatrixRename::accCanRls(uint32_t logreg_idx)
{
    assert(logreg_idx >= 4 && logreg_idx < 8);
    return accRlsEn[logreg_idx - 4];
}

//True means this acc can be rellocated a new acc prf later.
void MatrixRename::setAcc(uint32_t logreg_idx, bool mzero)
{
    assert(logreg_idx >= 4 && logreg_idx < 8);
    accRlsEn[logreg_idx - 4] = mzero;
}

void MatrixRename::regStats()
{
    SimObject::regStats();
    TileRegUse
    .name(name() + ".numRegisterUse")
    .init(tilePRFs)
    .desc("Count the register use number");
    TileRAT_read
    .name(name() + ".numRATRead")
    .desc("Count the number of RAT read");
    TileRAT_write
    .name(name() + ".numRATWrite")
    .desc("Count the number of RAT write");
    TileFreeList_read
    .name(name() + ".numFreeListRead")
    .desc("Count the number of FreeList read");
    TileFreeList_write
    .name(name() + ".numFreeListWrite")
    .desc("Count the number of FreeList write");

    AccRegUse
    .name(name() + ".numAccRegisterUse")
    .init(accPRFs)
    .desc("Count the Acc register use number");
    AccRAT_read
    .name(name() + ".numAccRATRead")
    .desc("Count the number of Acc RAT read");
    AccRAT_write
    .name(name() + ".numAccRATWrite")
    .desc("Count the number of Acc RAT write");
    AccFreeList_read    
    .name(name() + ".numAccFreeListRead")
    .desc("Count the number of Acc FreeList read");
}

// bool MatrixRename::freeList_empty()
// {
//     if(Matrix_FreeList.size()==0){
//         return true;
//     } else{
//         return false;
//     }
// }

bool MatrixRename::freeAccList_empty()
{
    if(Matrix_FreeAccList.size()==0){
        return true;
    } else{
        return false;
    }
}

bool MatrixRename::freeTileList_empty()
{
    if(Matrix_FreeTileList.size()==0){
        return true;
    } else{
        return false;
    }
}

// uint32_t MatrixRename::freeList_size()
// {
//     return Matrix_FreeList.size();
// }
uint32_t MatrixRename::freeTileList_size()
{
    return Matrix_FreeTileList.size();
}

uint32_t MatrixRename::freeAccList_size()
{
    return Matrix_FreeAccList.size();
}

// uint32_t MatrixRename::get_freeReg()
// {
//     assert(!(Matrix_FreeList.size()==0));
//     uint32_t freeReg;
//     freeReg = Matrix_FreeList.front();
//     std::cout << "Size before pop: " << Matrix_FreeList.size() << std::endl;
//     Matrix_FreeList.pop_front();
//     std::cout << "Size before pop: " << Matrix_FreeList.size() << std::endl;
//     RegUse[freeReg]++;
//     FreeList_read++;
//     return freeReg;
// }
uint32_t MatrixRename::get_freeTileReg()
{
    assert(!(Matrix_FreeTileList.size()==0));
    uint32_t freeReg;
    freeReg = Matrix_FreeTileList.front();
    std::cout << "Tile RAT Size before pop: " << Matrix_FreeTileList.size() << std::endl;
    Matrix_FreeTileList.pop_front();
    std::cout << "Tile RAT Size after pop: " << Matrix_FreeTileList.size() << std::endl;
    TileRegUse[freeReg]++;
    TileFreeList_read++;
    return freeReg;
}
uint32_t MatrixRename::get_freeAccReg()
{
    assert(!(Matrix_FreeAccList.size()==0));
    uint32_t freeReg;
    freeReg = Matrix_FreeAccList.front();
    std::cout << "ACC RAT Size before pop: " << Matrix_FreeAccList.size() << std::endl;
    Matrix_FreeAccList.pop_front();
    std::cout << "ACC RAT Size after pop: " << Matrix_FreeAccList.size() << std::endl;
    AccRegUse[freeReg-tilePRFs]++;
    AccFreeList_read++;
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
    if(phyreg_idx < tilePRFs){
        TileFreeList_write++;
        Matrix_FreeTileList.push_back(phyreg_idx);
        DPRINTF(MatrixRename, "Free Tile PR:%d\n", phyreg_idx);
    } else if (phyreg_idx < numPhysicalRegs && phyreg_idx >= tilePRFs){
        AccFreeList_write++;
        Matrix_FreeAccList.push_back(phyreg_idx);
        DPRINTF(MatrixRename, "Free Acc PR:%d\n", phyreg_idx);
    }
    // //
    // FreeList_write++;
    // Matrix_FreeList.push_back(phyreg_idx);
}

uint32_t MatrixRename::get_preg_RAT(uint32_t logreg_idx)
{
    if(logreg_idx < 4){
        assert((RegAliasTable[logreg_idx] < tilePRFs) || (RegAliasTable[logreg_idx] == 99));
        TileRAT_read++;
    } else if (logreg_idx < numLogicalRegs && logreg_idx >= 4) {
        AccRAT_read++;
        assert((RegAliasTable[logreg_idx] < numPhysicalRegs && RegAliasTable[logreg_idx] >= tilePRFs)|| (RegAliasTable[logreg_idx] == 99));
    }
    return RegAliasTable[logreg_idx];
}

void MatrixRename::set_preg_RAT(uint32_t logreg_idx, uint32_t phyreg_idx)
{
    RegAliasTable[logreg_idx] = phyreg_idx;
    set_PR_vld(phyreg_idx, false);
    if(logreg_idx < 4)
        TileRAT_write++;
    else if (logreg_idx < numLogicalRegs && logreg_idx >= 4){
        AccRAT_write++;
        accRlsEn[logreg_idx-4] = false;
    } 
    // RAT_write++;
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
    DPRINTF(MatrixRename, "%u %u %u %u %u %u %u %u %u %u %u %u %u %u %u %u %u %u %u %u %u %u %u %u %u %u %u %u %u %u %u %u\n"
        ,PR_vld[0],PR_vld[1],PR_vld[2],PR_vld[3],
        PR_vld[4],PR_vld[5],PR_vld[6],PR_vld[7],
        PR_vld[8], PR_vld[9], PR_vld[10], PR_vld[11],
        PR_vld[12], PR_vld[13], PR_vld[14], PR_vld[15], 
        PR_vld[16],PR_vld[17],PR_vld[18],PR_vld[19],
        PR_vld[20],PR_vld[21],PR_vld[22],PR_vld[23],
        PR_vld[24], PR_vld[25], PR_vld[26], PR_vld[27],
        PR_vld[28], PR_vld[29], PR_vld[30], PR_vld[31]);
}

} // namespace gem5