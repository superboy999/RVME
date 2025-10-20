/*
 * @Author: Chen WanQi
 * @Date: 2024-05-31 19:59:48
 * @LastEditTime: 2025-10-15 18:56:39
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
    ClockedObject(params), tileReg_num(params.tileReg_num), accReg_num(params.accReg_num),
    regWidth(params.regWidth), bank_num(params.bank_num), bank_depth(params.bank_depth)
{
    // physical_Mreg_256 = {};
    physical_tileMreg_128.resize(tileReg_num);
    physical_tileMreg_256.resize(tileReg_num);
    physical_tileMreg_512.resize(tileReg_num);
    physical_accMreg_128.resize(accReg_num);
    physical_accMreg_256.resize(accReg_num);
    physical_accMreg_512.resize(accReg_num);
    PR_owner.resize(tileReg_num + accReg_num, false);

    for (uint8_t i = 0; i < tileReg_num + accReg_num; i++){
        PR_owner[i] = false;
    }

    wtport.resize(4, false);
    rdport.resize(4, false);
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
    uint32_t _bankgrp, _bank, _entry, _byte_offset; 
    if(regWidth == 128){
        physical_tileMreg_128[phy_idx][bank_idx][entry][byte_offset] = data;
        DPRINTF(MatrixRF, "write %u to MatrixRF, phy_idx = %u, bank_idx = %u, entry = %u, byte_offset = %u\n", data, phy_idx, bank_idx, entry, byte_offset);
    } else if(regWidth == 256){
        if(phy_idx < 16){
            _bankgrp = phy_idx % 4;
            _bank = bank_idx + (phy_idx / 4);
            _entry = (phy_idx / 4) * 8 + entry;
            _byte_offset = byte_offset;
        } else {
            panic("phy_idx exceed the max number of tile register!");
        }
        physical_tileMreg_256[_bankgrp][_bank][_entry][_byte_offset] = data;
        DPRINTF(MatrixRF, "write %u to MatrixRF, phy_idx = %u, bank_group = %u, bank_idx = %u, entry = %u, byte_offset = %u\n", data, phy_idx, _bankgrp, _bank, _entry, _byte_offset);
    } else if(regWidth == 512){
        physical_tileMreg_512[phy_idx][bank_idx][entry][byte_offset] = data;
        DPRINTF(MatrixRF, "write %u to MatrixRF, phy_idx = %u, bank_idx = %u, entry = %u, byte_offset = %u\n", data, phy_idx, bank_idx, entry, byte_offset);
    } else {
        panic("Wrong register width configuration!");
    }
    numwrites_byte++;
}

uint8_t MatrixRF::rdreg_byte(uint32_t phy_idx, uint32_t bank_idx, uint32_t entry, uint32_t byte_offset)
{
    uint32_t _bankgrp, _bank, _entry, _byte_offset; 
    if(regWidth == 128){
        DPRINTF(MatrixRF, "read %u from MatrixRF, phy_idx = %u, bank_idx = %u, entry = %u, byte_offset = %u\n", physical_tileMreg_128[phy_idx][bank_idx][entry][byte_offset], phy_idx, bank_idx, entry, byte_offset);
        return physical_tileMreg_128[phy_idx][bank_idx][entry][byte_offset];
    } else if(regWidth == 256){
        if(phy_idx < 16){
            _bankgrp = phy_idx % 4;
            _bank = bank_idx + (phy_idx / 4);
            _entry = (phy_idx / 4) * 8 + entry;
            _byte_offset = byte_offset;
        } else {
            panic("phy_idx exceed the max number of tile register!");
        }
        DPRINTF(MatrixRF, "read %u from MatrixRF, phy_idx = %u, bank_group = %u, bank_idx = %u, entry = %u, byte_offset = %u\n", physical_tileMreg_256[_bankgrp][_bank][_entry][_byte_offset], phy_idx, _bankgrp, _bank, _entry, _byte_offset);
        return physical_tileMreg_256[_bankgrp][_bank][_entry][_byte_offset];
    } else if(regWidth == 512){
        DPRINTF(MatrixRF, "read %u from MatrixRF, phy_idx = %u, bank_group = %u, bank_idx = %u, entry = %u, byte_offset = %u\n", physical_tileMreg_512[phy_idx][bank_idx][entry][byte_offset], phy_idx, _bankgrp, _bank, _entry, _byte_offset);
        return physical_tileMreg_512[phy_idx][bank_idx][entry][byte_offset];
    } else {
        panic("Wrong register width configuration!");
    }
    numreads_byte++;
}

uint32_t MatrixRF::rdreg_int32(uint32_t phy_idx, uint32_t bank_idx, uint32_t entry, uint32_t word_offset)
{
    uint32_t _bankgrp, _bank, _entry, _word_offset; 
    if(regWidth == 128){
        panic("Unsupported register width!");
    } else if (regWidth == 256){
        if(phy_idx < 16){
            panic("phy_idx exceed the max number of the acc register!");
        } else if (phy_idx >= 16 && phy_idx < 32){
            _bankgrp = ((phy_idx-16) % 4 * 4 + 4 * entry + bank_idx) / 4 % 4;
            _bank = bank_idx;
            _entry = (phy_idx - 16)* 2 + entry / 4;
            _word_offset = (word_offset % 2) * 4;
        } else {
            panic("phy_idx exceed the max number of tile register!");
        }
        DPRINTF(MatrixRF, "read %u from MatrixRF, phy_idx = %u, bank group = %u, bank_idx = %u, entry = %u, word_offset = %u\n", 
            (static_cast<uint32_t>(physical_accMreg_256[_bankgrp][_bank][_entry][(_word_offset + 3)] << 24) |
            (static_cast<uint32_t>(physical_accMreg_256[_bankgrp][_bank][_entry][_word_offset + 2]) << 16) |
            (static_cast<uint32_t>(physical_accMreg_256[_bankgrp][_bank][_entry][_word_offset + 1]) << 8) |
            (static_cast<uint32_t>(physical_accMreg_256[_bankgrp][_bank][_entry][_word_offset + 0]) )),
            phy_idx, _bankgrp, _bank, _entry, _word_offset);
        return (static_cast<uint32_t>(physical_accMreg_256[_bankgrp][_bank][_entry][(_word_offset + 3)] << 24) |
            (static_cast<uint32_t>(physical_accMreg_256[_bankgrp][_bank][_entry][_word_offset + 2]) << 16) |
            (static_cast<uint32_t>(physical_accMreg_256[_bankgrp][_bank][_entry][_word_offset + 1]) << 8) |
            (static_cast<uint32_t>(physical_accMreg_256[_bankgrp][_bank][_entry][_word_offset + 0])));
    } else if (regWidth == 512){
        panic("Unsupported register width!");
    } else {
        panic("Wrong register width configuration!");
    }
    numreads_byte += 4;
}    

void MatrixRF::wtreg_int32(uint32_t phy_idx, uint32_t bank_idx, uint32_t entry, uint32_t word_offset, uint32_t data)
{
    uint32_t _bankgrp, _bank, _entry, _word_offset;
    if(regWidth == 128){
        panic("Unsupported register width!");
    } else if (regWidth == 256){
        if(phy_idx < 16){
            panic("phy_idx exceed the max number of the acc register!");
        } else if (phy_idx >= 16 && phy_idx < 32){
            _bankgrp = ((phy_idx-16) % 4 * 4 + 4 * entry + bank_idx) / 4 % 4;
            _bank = bank_idx;
            _entry = (phy_idx - 16)* 2 + entry / 4;
            _word_offset = (word_offset % 2) * 4;
        } else {
            panic("phy_idx exceed the max number of tile register!");
        }
        physical_accMreg_256[_bankgrp][_bank][_entry][_word_offset + 3] = static_cast<uint8_t>((data >> 24) & 0xFF);
        physical_accMreg_256[_bankgrp][_bank][_entry][_word_offset + 2] = static_cast<uint8_t>((data >> 16) & 0xFF);
        physical_accMreg_256[_bankgrp][_bank][_entry][_word_offset + 1] = static_cast<uint8_t>((data >> 8) & 0xFF);
        physical_accMreg_256[_bankgrp][_bank][_entry][_word_offset + 0] = static_cast<uint8_t>((data >> 0) & 0xFF);
        DPRINTF(MatrixRF, "write %u to MatrixRF, phy_idx = %u, bank_grp = %u, bank_idx = %u, entry = %u, word_offset = %u\n", data, phy_idx, _bankgrp, _bank, _entry, _word_offset);
    } else if (regWidth == 512){
        panic("Unsupported register width!");
    } else {
        panic("Wrong register width configuration!");
    }
    numwrites_byte += 4;
}

void MatrixRF::printRF(uint32_t phy_idx, uint32_t bank_idx, uint32_t entry, uint32_t word_offset) {
    // 确保 phy_idx 在有效范围内
    assert(phy_idx < tileReg_num + accReg_num);
    if(phy_idx < 16){
        // 遍历所有 bank, entry 和 byte_offset
        for (uint32_t bank_idx = 0; bank_idx < bank_num; ++bank_idx) {
            for (uint32_t entry = phy_idx / 4; entry < phy_idx / 4 + 8; ++entry) {
                std::cout << "Bank " << bank_idx << ", Entry " << entry << ": ";
                for (uint32_t byte_offset = 0; byte_offset < regWidth; ++byte_offset) {
                    // 打印指定 phy_idx, bank_idx, entry 和 byte_offset 的 data
                    std::cout << "Byte " << byte_offset << ": ";
                    std::cout << static_cast<unsigned>(physical_tileMreg_256[phy_idx % 4][bank_idx][entry][byte_offset]) << " ";
                }
                std::cout << std::endl;
            }
            std::cout << std::endl;
        }
    } else {
        for (uint8_t i = 0; i < 2; i++){
            for (uint8_t bankgrp = 0; bankgrp < 4; bankgrp++){
                for (uint8_t bank = 0; bank < 4; bank++){
                    std::cout << "Bank Group " << bankgrp << ", Bank " << bank << ", Entry " << entry + i * 8 << ": ";
                    for (uint8_t byte_offset = 0; byte_offset < regWidth; byte_offset++){
                        std::cout << "Byte " << byte_offset << ": ";
                        std::cout << static_cast<unsigned>(physical_accMreg_256[bankgrp][bank][(phy_idx-16) / 4 + i][byte_offset]) << " ";
                    }
                    std::cout << std::endl;
                }
            }
        }
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

bool MatrixRF::occupy_wtport(uint8_t phy_idx) //FIXME: add port idx, every bank has one wt port
{
    if(phy_idx < 16){
        uint8_t bankgrp = phy_idx % 4;
        if(!wtport[bankgrp]){
            wtport[bankgrp] = true;
            return true;
        } else{
            return false;
        }
    } else {
        panic("phy_idx exceed the max number of tile register!");
    }
    // if(!wtport[phy_idx / 4]){
    //     wtport[phy_idx / 4] = true;
    //     return true;
    // } else{
    //     return false;
    // }
}

bool MatrixRF::occupy_wtport(uint32_t phy_idx, uint32_t bank_idx, uint32_t entry) //FIXME: add port idx, every bank has one wt port
{
    if(phy_idx >= 16 && phy_idx < 32){
        uint8_t bankgrp = ((phy_idx-16) % 4 * 4 + 4 * entry + bank_idx) / 4 % 4;
        if(!wtport[bankgrp]){
            wtport[bankgrp] = true;
            return true;
        } else{
            return false;
        }
    } else {
        panic("phy_idx exceed the max number of tile register!");
    }

}
void MatrixRF::rls_wrport(uint8_t phy_idx)
{
    wtport[phy_idx % 4] = false;
}

void MatrixRF::rls_wrport(uint32_t phy_idx, uint32_t bank_idx, uint32_t entry)
{
    uint8_t bankgrp = ((phy_idx-16) % 4 * 4 + 4 * entry + bank_idx) / 4 % 4;
    wtport[bankgrp] = false;
}

bool MatrixRF::occupy_rdport(uint8_t phy_idx) // FIXME: add port idx, every bank has one rd port
{
    if(phy_idx < 16){
        uint8_t bankgrp = phy_idx % 4;
        if(!rdport[bankgrp]){
            rdport[bankgrp] = true;
            return true;
        } else{
            return false;
        }
    } else {
        panic("phy_idx exceed the max number of tile register!");
    }
}

bool MatrixRF::occupy_rdport(uint8_t phy_idx, uint32_t bank_idx, uint32_t entry) // FIXME: add port idx, every bank has one rd port
{
    if(phy_idx >= 16 && phy_idx < 32){
        uint8_t bankgrp = ((phy_idx-16) % 4 * 4 + 4 * entry + bank_idx) / 4 % 4;
        if(!rdport[bankgrp]){
            rdport[bankgrp] = true;
            return true;
        } else{
            return false;
        }
    } else {
        panic("phy_idx exceed the max number of tile register!");
    }
}

void MatrixRF::rls_rdport(uint8_t phy_idx)
{
    rdport[phy_idx % 4] = false;
}

void MatrixRF::rls_rdport(uint8_t phy_idx, uint32_t bank_idx, uint32_t entry)
{
    uint8_t bankgrp = ((phy_idx-16) % 4 * 4 + 4 * entry + bank_idx) / 4 % 4;
    rdport[bankgrp] = false;
}

} //namespace gem5