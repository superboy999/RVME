/*
 * @Author: superboy
 * @Date: 2024-07-08 16:43:16
 * @LastEditTime: 2025-10-17 19:12:21
 * @LastEditors: superboy
 * @Description: 
 * @FilePath: /gem5-rvm/src/cpu/matrix_engine/rename/matrix_rename.hh
 * 
 */
#ifndef __CPU_MATRIX_RENAME_HH__
#define __CPU_MATRIX_RENAME_HH__

#include <cstdint>
#include <deque>

#include "params/MatrixRename.hh"
#include "sim/sim_object.hh"
#include "base/statistics.hh"

namespace gem5
{
struct MatrixRenameParams;
class MatrixRename : public SimObject
{
public:
    MatrixRename(const MatrixRenameParams &params);
    ~MatrixRename();

    void regStats() override;
    bool freeTileList_empty();
    bool freeAccList_empty();
    uint32_t freeTileList_size();
    uint32_t freeAccList_size();
    uint32_t get_freeTileReg();
    uint32_t get_freeAccReg();
    void set_freeReg(uint32_t phyreg_idx);
    uint32_t get_preg_RAT(uint32_t logreg_idx);
    void set_preg_RAT(uint32_t logreg_idx, uint32_t phyreg_idx);

    bool get_PR_vld(uint8_t phyreg_idx);
    void set_PR_vld(uint8_t phyreg_idx, bool vld);

    void print_RAT();
    void print_pr_vld();

    void regLock(uint32_t phyreg_idx);
    void regrls(uint32_t phyreg_idx);
    bool checkLock(uint32_t phyreg_idx);

    bool accCanRls(uint32_t logreg_idx); // just for mzero, and this is to change the Acc, forgive the former value of acc
    void setAcc(uint32_t logreg_idx, bool mzero);
    std::vector<bool> accInitial; //indicate the acc initial status
private:
    //python configuration
    const uint32_t numPhysicalRegs;
    const uint32_t numLogicalRegs;
    const uint32_t tilePRFs;
    const uint32_t accPRFs;
    //Free List RegisterFile(Just save the number of the physical registerfile)
    // std::deque<uint32_t> Matrix_FreeList; //This queue naturally keep the order of
    std::deque<uint32_t> Matrix_FreeAccList; //This queue naturally keep the order of
    std::deque<uint32_t> Matrix_FreeTileList; //This queue naturally keep the order of
    // the usage of physical registers
    uint32_t RegAliasTable[8];
    std::vector<bool> PR_vld;//Indicate this value is writen-back/ready; Change to physical flag
    std::vector<uint32_t> PR_using;// Fix bug of free the using register!
    std::vector<bool> accRlsEn;
public:
    statistics::Vector TileRegUse;
    statistics::Scalar TileRAT_read;
    statistics::Scalar TileRAT_write;
    statistics::Scalar TileFreeList_read;
    statistics::Scalar TileFreeList_write;

    statistics::Vector AccRegUse;
    statistics::Scalar AccRAT_read;
    statistics::Scalar AccRAT_write;
    statistics::Scalar AccFreeList_read;
    statistics::Scalar AccFreeList_write;
};


} //namespace gem5

#endif