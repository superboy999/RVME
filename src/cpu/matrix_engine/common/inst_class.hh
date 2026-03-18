/*
 * @Author: superboy
 * @Date: 2025-03-17 01:19:14
 * @LastEditTime: 2025-10-18 20:05:01
 * @LastEditors: superboy
 * @Description: 
 * @FilePath: /gem5-rvm/src/cpu/matrix_engine/common/inst_class.hh
 * 
 */
#ifndef __CPU_MATRIX_INST_CLASS_HH__
#define __CPU_MATRIX_INST_CLASS_HH__

#include <bitset>
#include <cassert>
#include <cstdint>

namespace gem5
{

class MatrixInstClass
{
public:
    enum InstType
    {
        MultiplyAndAccumulate,
        Move,
        Add,
        Store,
        Load
    };
    enum DataType
    {
        INT8,
        INT16,
        INT32
    };
    struct MatrixSize
    {
        uint16_t partial_size;
        uint8_t num_column;
        uint8_t num_row;
    };

    enum EW_OP
    {
        MADD,
        MSUB,
        MMUL,
        MMULH,
        MMAX,
        MUMAX,
        MMIN,
        MUMIN,
        MSRL,
        MSLL,
        MSRA,
        MLUT
    };
    // struct MatrixRLEN
    // {
    //     std::bitset<matrix_RLEN> bits;
    // };
    // struct MatrixRowSize
    // {
    //     std::bitset<matrix_RLEN/32> bits;
    // };
    void load()
    {
        load_cnt = load_cnt + 1;
    }

    void execute()
    {
        exe_num = exe_num + 1;
    }

public:

    // uint64_t matrix_RLEN = 512;
    bool isSigned;
    MatrixSize matrix_size;
    InstType inst_type;
    DataType data_type;
    uint8_t xreg_idx;
    uint8_t yreg_idx;
    uint8_t zreg_idx;
    //
    bool load_done;
    uint32_t load_cnt;
    //
    bool executed;
    uint32_t exe_num;

    bool instDone = false; //only for lane to check if this instruction is done!

    uint8_t lane_idx; //record which lane is executing this inst
    uint8_t ewu_idx; //record which ewu is executing this inst

    void lane_reset(){
        load_cnt = 0;
        exe_num = 0;
        load_done = false;
        executed = false;
        instDone = false;
    }
private:
    uint64_t matrix_RLEN = 256; //FIXME: Maybe be can be configured by python later!!
};
} // namespace gem5


#endif