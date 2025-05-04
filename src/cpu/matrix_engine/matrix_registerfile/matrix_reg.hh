/*
 * @Author: superboy
 * @Date: 2024-06-03 23:09:30
 * @LastEditTime: 2024-09-18 14:47:15
 * @LastEditors: superboy
 * @Description: 
 * @FilePath: /gem5-rvm/src/cpu/matrix_engine/matrix_registerfile/matrix_reg.hh
 * 
 */

#ifndef __CPU_MATRIX_REG_HH__
#define __CPU_MATRIX_REG_HH__

#include <cassert>
#include <cstdint>
#include <array>
#include <string>
#include <vector>
#include <queue>
#include <iostream>

#include "sim/clocked_object.hh"
#include "params/MatrixRF.hh"
#include "base/statistics.hh"

namespace gem5
{

struct MatrixRFParams;

class MatrixRF : public ClockedObject
{
public:
    MatrixRF(const MatrixRFParams &params);
    ~MatrixRF();

    void regStats() override;
    void wtreg_byte(uint32_t phy_idx, uint32_t bank_idx, uint32_t entry, uint32_t byte_offset, uint8_t data);
    uint8_t rdreg_byte(uint32_t phy_idx, uint32_t bank_idx, uint32_t entry, uint32_t byte_offset);
    void printRF(uint32_t phy_idx);

    bool try_occupy(uint32_t phy_id);
    bool check_status(uint32_t phy_id);
    void rls(uint32_t phy_id); 

    bool occupy_wtport();
    void rls_wrport();
public: //normally this should be private, but for access convenient
    //python configuration
    uint32_t physicReg_num;
    uint32_t regWidth;
    uint32_t bank_num;
    uint32_t bank_depth;
private:
    // entrywidth/entrydepth/banknum/phyRF
    // std::array<std::array<std::array<std::array<uint8_t, 4>, 4>, 4>, 16> physical_Mreg_128{};
    // std::array<std::array<std::array<std::array<uint8_t, 8>, 8>, 4>, 16> physical_Mreg_256{};
    // std::array<std::array<std::array<std::array<uint8_t, 16>, 16>, 4>, 16> physical_Mreg_512{};
    std::vector<std::array<std::array<std::array<uint8_t, 4>, 4>, 4>> physical_Mreg_128;
    std::vector<std::array<std::array<std::array<uint8_t, 8>, 8>, 4>> physical_Mreg_256;
    std::vector<std::array<std::array<std::array<uint8_t, 16>, 16>, 4>> physical_Mreg_512;
    // add new feature 24/7/27
    std::vector<bool>  PR_owner; //false means free, true means occupied!
    bool wtport; // false means free
    // std::vector<std::deque<std::array<uint8_t, 8>>> phy_reg[8];
    // std::vector<std::vector<std::array<uint8_t, 4>>> phy_reg[8];
public:
    statistics::Scalar numreads_byte;
    statistics::Scalar numwrites_byte;

    // enum RegWidthStatus
    // {
    //     SHORT128,
    //     MID256,
    //     LONG512
    // };
    
};
} //namespace gem5

#endif