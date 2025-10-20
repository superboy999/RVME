/*
 * @Author: superboy
 * @Date: 2024-07-08 15:57:02
 * @LastEditTime: 2025-10-18 00:39:30
 * @LastEditors: superboy
 * @Description: 
 * @FilePath: /gem5-rvm/src/cpu/matrix_engine/reorder_buffer/reorder_buffer.hh
 * 
 */
#ifndef __CPU_MATRIX_ROB_HH__
#define __CPU_MATRIX_ROB_HH__

#include <cstdint>
#include <cassert>
#include <vector>
#include <array>

#include "params/ReorderBuffer.hh"
#include "sim/ticked_object.hh"
#include "cpu/matrix_engine/matrix_engine.hh"
#include "base/statistics.hh"

namespace gem5
{
class MatrixEngine;
struct ReorderBufferParams;
class ReorderBuffer : public TickedObject
{
public:
    class rob_entry{
        public:
        rob_entry(uint32_t logic_idx, uint32_t physic_idx, uint32_t old_dst):
        logic_idx(logic_idx), physic_idx(physic_idx), old_dst(old_dst), old_dst_vld(false), executed(false),
        valid(false){}
        ~rob_entry(){}

        uint32_t logic_idx;
        uint32_t physic_idx;
        uint32_t old_dst; //记录了上一个RAT的logic_idx对应的physic_idx,用于保守的手段更新freelist
        bool old_dst_vld;
        bool valid;
        bool executed;
    };

    class memdep_entry{
        public:
        memdep_entry():
        valid(false), read_num(0){}
        ~memdep_entry(){}
        bool valid;
        uint32_t read_num; //这样可以保证在所有后面所有需要的指令都读完这个才能换下一个entry, 好像不是很需要？
    };

    //FIXME:在dispatchGrant里面没有这个full的判断条件
    std::array<std::vector<memdep_entry>, 8> memdep_table; //For 8 logic RF, only record RAW dependency, vector的entry个数表示某个logic RF被wt了多少次，而每个entry对应的数字表示被多少个指令读了
    std::array<uint32_t, 8> mdtail = {0}; //memory dependency table tail pointer
    std::array<uint32_t, 8> mdhead = {0}; //memory dependency table head pointer
    std::array<uint32_t, 8> mdvalid_element = {0};

    ReorderBuffer(const ReorderBufferParams &params);
    ~ReorderBuffer();

    void set_matrixEnginePtr(MatrixEngine* _matrix_engine);
    void startTicking();
    void stopTicking();
    bool isOccupied();
    void regStats() override;
    void evaluate() override;

    bool rob_full();
    bool rob_empty();

    bool md_full(uint8_t idx); // all inputs are logic register index
    uint32_t set_md_entry(uint8_t idx); // only record wt opera
    void set_md_entry_valid(uint8_t idx, uint32_t entry_idx);
    bool raw_solved(uint8_t idx, uint32_t check_tail) {return mdhead[idx] == check_tail;}
    bool macc_raw_solved(uint8_t idx, uint32_t check_tail) {return mdhead[idx] + 1 == check_tail;}

    uint32_t set_rob_entry(uint32_t logic_idx, uint32_t physic_idx, uint32_t old_dst, bool old_dst_vld); 
    void set_rob_entry_executed(uint32_t idx); // This idx is used to index the entry number of ROB.
    uint32_t meminst_num = 0;
    bool meminst_Empty() { return meminst_num == 0; }
private:
    bool Occupied;

    // python configuration
    const uint32_t ROB_depth;
    const uint32_t MDU_depth;

    std::vector<rob_entry *> rob;
    uint32_t tail;
    uint32_t head;
    uint32_t valid_element;
public:
    statistics::Scalar MatrixROBentryUsed;
    statistics::Scalar ROB_read;
    statistics::Scalar ROB_write;
private:
    MatrixEngine* matrix_engine;
};

} // namespace gem5


#endif  