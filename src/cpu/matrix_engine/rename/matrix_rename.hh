/*
 * @Author: superboy
 * @Date: 2024-07-08 16:43:16
 * @LastEditTime: 2025-03-26 01:18:48
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
    bool freeList_empty();
    uint32_t freeList_size();
    uint32_t get_freeReg();
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
private:
    //python configuration
    const uint32_t numPhysicalRegs;
    const uint32_t numLogicalRegs;

    //Free List RegisterFile(Just save the number of the physical registerfile)
    std::deque<uint32_t> Matrix_FreeList; //This queue naturally keep the order of the usage of physical register
    uint32_t RegAliasTable[8];
    std::vector<bool> PR_vld;//Indicate this value is writen-back/ready; Change to physical flag
    std::vector<uint32_t> PR_using;// Fix bug of free the using register!
public:
    statistics::Vector RegUse;
    statistics::Scalar RAT_read;
    statistics::Scalar RAT_write;
    statistics::Scalar FreeList_read;
    statistics::Scalar FreeList_write;
};


} //namespace gem5

#endif