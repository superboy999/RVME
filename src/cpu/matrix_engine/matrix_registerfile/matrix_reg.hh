/*
 * @Author: superboy
 * @Date: 2024-06-03 23:09:30
 * @LastEditTime: 2025-10-23 14:44:35
 * @LastEditors: superboy
 * @Description: 
 * @FilePath: /SJTU-matrix-engine/src/cpu/matrix_engine/matrix_registerfile/matrix_reg.hh
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
    uint32_t rdreg_int32(uint32_t phy_idx, uint32_t bank_idx, uint32_t entry, uint32_t word_offset);
    void wtreg_int32(uint32_t phy_idx, uint32_t bank_idx, uint32_t entry, uint32_t word_offset, uint32_t data);
    void printRF(uint32_t phy_idx, uint32_t bank_idx, uint32_t entry, uint32_t word_offset);

    void wt_A_4col(uint32_t phy_idx, uint32_t bank_idx, uint32_t entry, uint8_t* data, uint8_t size);
    uint8_t* rd_A_4col(uint32_t phy_idx, uint32_t bank_idx, uint32_t entry, uint8_t size);
    void wt_Brow(uint32_t phy_idx, uint32_t bank_idx, uint32_t entry, uint8_t* data, uint8_t size);
    uint8_t* rd_Brow(uint32_t phy_idx, uint32_t bank_idx, uint32_t entry, uint8_t size);
    
    void wt_Crow(uint32_t phy_idx, uint32_t bank_idx, uint32_t entry, uint32_t word_offset, uint8_t* data, uint8_t size);
    void wt_Ctrow(uint32_t phy_idx, uint32_t bank_idx, uint32_t entry, uint32_t word_offset, uint8_t* data, uint8_t size);
    uint8_t* rd_Crow(uint32_t phy_idx, uint32_t bank_idx, uint32_t entry, uint32_t word_offset, uint8_t size);
    uint8_t* rd_Ctrow(uint32_t phy_idx, uint32_t bank_idx, uint32_t entry, uint32_t word_offset, uint8_t size);
    void set_reg_zero(uint32_t phy_idx);

    bool try_occupy(uint32_t phy_id);
    bool check_status(uint32_t phy_id);
    void rls(uint32_t phy_id); 

    bool occupy_wtport(uint8_t phy_idx); //for tile reg, because tile reg are allocate in one bank group(4 banks), so it will not change during one read/write with one prf
    bool wtport_valid(uint32_t phy_idx, uint32_t bank_idx, uint32_t entry); // for acc reg, every row has interleaved in 4 bank groups, so it will changed occupy group during one read/write.
    bool occupy_wtport(uint32_t phy_idx, uint32_t bank_idx, uint32_t entry); // for acc reg, every row has interleaved in 4 bank groups, so it will changed occupy group during one read/write.
    void rls_wrport(uint8_t phy_idx);
    void rls_wrport(uint32_t phy_idx, uint32_t bank_idx, uint32_t entry); // for acc, every write for one row has to release once. cause next write will be in another bank group.

    bool occupy_rdport(uint8_t phy_idx);
    bool occupy_rdport(uint8_t phy_idx, uint32_t bank_idx, uint32_t entry);
    bool rdport_valid(uint8_t phy_idx, uint32_t bank_idx, uint32_t entry);
    void rls_rdport(uint8_t phy_idx);
    void rls_rdport(uint8_t phy_idx, uint32_t bank_idx, uint32_t entry);
public: //normally this should be private, but for access convenient
    //python configuration
    uint32_t tileReg_num;
    uint32_t accReg_num;
    uint32_t regWidth;
    uint32_t bank_num;
    uint32_t bank_depth;
private:
    // entrywidth/entrydepth/banknum/phyRF
    // std::array<std::array<std::array<std::array<uint8_t, 4>, 4>, 4>, 16> physical_Mreg_128{};
    // std::array<std::array<std::array<std::array<uint8_t, 8>, 8>, 4>, 16> physical_Mreg_256{};
    // std::array<std::array<std::array<std::array<uint8_t, 16>, 16>, 4>, 16> physical_Mreg_512{};
    // This is interesting, array defines the fix size of one register, however vector defines the number of regs+
    std::vector<std::array<std::array<std::array<uint8_t, 4>, 4>, 4>> physical_tileMreg_128;
    std::vector<std::array<std::array<std::array<uint8_t, 8>, 32>, 4>> physical_tileMreg_256;
    std::vector<std::array<std::array<std::array<uint8_t, 16>, 16>, 4>> physical_tileMreg_512;

    std::vector<std::array<std::array<std::array<uint8_t, 4>, 4>, 4>> physical_accMreg_128;
    std::vector<std::array<std::array<std::array<uint8_t, 8>, 32>, 4>> physical_accMreg_256;
    std::vector<std::array<std::array<std::array<uint8_t, 16>, 16>, 4>> physical_accMreg_512;
    // add new feature 24/7/27
    std::vector<bool> PR_owner; //false means free, true means occupied!
    std::vector<bool> wtport; // false means free, take four banks(one bank group) as a unit
    std::vector<bool> rdport; // false means free
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