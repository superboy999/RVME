/*
 * @Author: Chen WanQi
 * @Date: 2024-05-31 19:59:48
 * @LastEditTime: 2024-09-18 14:57:53
 * @LastEditors: superboy
 * @Description: 
 * @FilePath: /gem5-rvm/src/cpu/matrix_engine/matrix_registerfile/matrix_reg.cc
 * 
 */
#include "cpu/matrix_engine/matrix_registerfile/matrix_reg.hh"
#include "debug/MatrixRF.hh"

#include <cstdint>
#include <cassert>

namespace gem5
{
MatrixRF::MatrixRF(const MatrixRFParams &params):
    ClockedObject(params), physicReg_num(params.physicReg_num),
    regWidth(params.regWidth), bank_num(params.bank_num), bank_depth(params.bank_depth)
{
    // physical_Mreg_256 = {};
    physical_Mreg_128.resize(physicReg_num);
    physical_Mreg_256.resize(physicReg_num);
    physical_Mreg_512.resize(physicReg_num);
    PR_owner.resize(physicReg_num, false);

    for (uint8_t i = 0; i < physicReg_num; i++){
        PR_owner[i] = false;
    }

    wtport = false;
}

MatrixRF::~MatrixRF()
{}

void MatrixRF::regStats()
{
    ClockedObject::regStats();
    numreads_byte
        .name(name() + ".numreadsbyte")
        .desc("Count how many times bytes reads from regfile");
    numwrites_byte
        .name(name() + ".numwritesbyte")
        .desc("Count how many times bytes writes from regfile");
}



void MatrixRF::wtreg_byte(uint32_t phy_idx, uint32_t bank_idx, uint32_t entry, uint32_t byte_offset, uint8_t data)
{
    if(regWidth == 128){
        physical_Mreg_128[phy_idx][bank_idx][entry][byte_offset] = data;
        DPRINTF(MatrixRF, "write %u to MatrixRF, phy_idx = %u, bank_idx = %u, entry = %u, byte_offset = %u\n", data, phy_idx, bank_idx, entry, byte_offset);
    } else if(regWidth == 256){
        physical_Mreg_256[phy_idx][bank_idx][entry][byte_offset] = data;
        DPRINTF(MatrixRF, "write %u to MatrixRF, phy_idx = %u, bank_idx = %u, entry = %u, byte_offset = %u\n", data, phy_idx, bank_idx, entry, byte_offset);
    } else if(regWidth == 512){
        physical_Mreg_512[phy_idx][bank_idx][entry][byte_offset] = data;
        DPRINTF(MatrixRF, "write %u to MatrixRF, phy_idx = %u, bank_idx = %u, entry = %u, byte_offset = %u\n", data, phy_idx, bank_idx, entry, byte_offset);
    } else {
        panic("Wrong register width configuration!");
    }
    numwrites_byte++;
}

uint8_t MatrixRF::rdreg_byte(uint32_t phy_idx, uint32_t bank_idx, uint32_t entry, uint32_t byte_offset)
{
    if(regWidth == 128){
        DPRINTF(MatrixRF, "read %u from MatrixRF, phy_idx = %u, bank_idx = %u, entry = %u, byte_offset = %u\n", physical_Mreg_128[phy_idx][bank_idx][entry][byte_offset], phy_idx, bank_idx, entry, byte_offset);
        return physical_Mreg_128[phy_idx][bank_idx][entry][byte_offset];
    } else if(regWidth == 256){
        DPRINTF(MatrixRF, "read %u from MatrixRF, phy_idx = %u, bank_idx = %u, entry = %u, byte_offset = %u\n", physical_Mreg_256[phy_idx][bank_idx][entry][byte_offset], phy_idx, bank_idx, entry, byte_offset);
        return physical_Mreg_256[phy_idx][bank_idx][entry][byte_offset];
    } else if(regWidth == 512){
        DPRINTF(MatrixRF, "read %u from MatrixRF, phy_idx = %u, bank_idx = %u, entry = %u, byte_offset = %u\n", physical_Mreg_512[phy_idx][bank_idx][entry][byte_offset], phy_idx, bank_idx, entry, byte_offset);
        return physical_Mreg_512[phy_idx][bank_idx][entry][byte_offset];
    } else {
        panic("Wrong register width configuration!");
    }
    numreads_byte++;
}

void MatrixRF::printRF(uint32_t phy_idx) {
    // 确保 phy_idx 在有效范围内
    assert(phy_idx < physicReg_num);

    // 遍历所有 bank, entry 和 byte_offset
    for (uint32_t bank_idx = 0; bank_idx < bank_num; ++bank_idx) {
        for (uint32_t entry = 0; entry < bank_depth; ++entry) {
            std::cout << "Bank " << bank_idx << ", Entry " << entry << ": ";
            for (uint32_t byte_offset = 0; byte_offset < regWidth; ++byte_offset) {
                // 打印指定 phy_idx, bank_idx, entry 和 byte_offset 的 data
                std::cout << "Byte " << byte_offset << ": ";
                std::cout << static_cast<unsigned>(physical_Mreg_256[phy_idx][bank_idx][entry][byte_offset]) << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
}

// used in sperate SRAM bank====================================
// bool MatrixRF::try_occupy(uint32_t phy_id)
// {
//     if(PR_owner[phy_id] == false){
//         DPRINTF(MatrixRF, "Occupied %d successful!\n", phy_id);
//         PR_owner[phy_id] = true;
//         return true;
//     } else {
//         DPRINTF(MatrixRF, "Occupied %d fail!\n", phy_id);
//         return false;
//     }
// }

// bool MatrixRF::check_status(uint32_t phy_id)
// {
//     return PR_owner[phy_id];
// }

// void MatrixRF::rls(uint32_t phy_id)
// {
//     assert(PR_owner[phy_id] == true);
//     DPRINTF(MatrixRF, "release %d successful!\n", phy_id);
//     PR_owner[phy_id] = false;
// }
// 暂时注释掉====================================

// used in integrated SRAM bank====================================
// bool MatrixRF::try_occupy(uint32_t phy_id)
// {
//     if(PR_owner[0] == false){
//         DPRINTF(MatrixRF, "Occupied %d successful!\n", phy_id);
//         PR_owner[0] = true;
//         return true;
//     } else {
//         DPRINTF(MatrixRF, "Occupied %d fail!\n", phy_id);
//         return false;
//     }
// }

// bool MatrixRF::check_status(uint32_t phy_id)
// {
//     return PR_owner[0];
// }

// void MatrixRF::rls(uint32_t phy_id)
// {
//     assert(PR_owner[0] == true);
//     DPRINTF(MatrixRF, "release %d successful!\n", phy_id);
//     PR_owner[0] = false;
// }
// 暂时注释掉====================================
bool MatrixRF::try_occupy(uint32_t phy_id)
{
    return true;
}

bool MatrixRF::check_status(uint32_t phy_id)
{
    return false;
}

void MatrixRF::rls(uint32_t phy_id)
{
    DPRINTF(MatrixRF, "release %d successful!\n", phy_id);
}

bool MatrixRF::occupy_wtport()
{
    if(!wtport){
        wtport = true;
        return true;
    } else{
        return false;
    }
}

void MatrixRF::rls_wrport()
{
    wtport = false;
}
} //namespace gem5