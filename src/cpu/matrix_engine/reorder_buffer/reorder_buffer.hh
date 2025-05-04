/*
 * @Author: superboy
 * @Date: 2024-07-08 15:57:02
 * @LastEditTime: 2025-03-27 00:04:32
 * @LastEditors: superboy
 * @Description: 
 * @FilePath: /gem5-rvm/src/cpu/matrix_engine/reorder_buffer/reorder_buffer.hh
 * 
 */
#ifndef __CPU_MATRIX_ROB_HH__
#define __CPU_MATRIX_ROB_HH__

#include <cstdint>
#include <cassert>

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

    uint32_t set_rob_entry(uint32_t logic_idx, uint32_t physic_idx, uint32_t old_dst, bool old_dst_vld); 
    void set_rob_entry_executed(uint32_t idx); // This idx is used to index the entry number of ROB.
private:
    bool Occupied;

    // python configuration
    const uint32_t ROB_depth;

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