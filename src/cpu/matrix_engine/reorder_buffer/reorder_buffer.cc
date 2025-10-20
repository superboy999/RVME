#include "cpu/matrix_engine/reorder_buffer/reorder_buffer.hh"
#include "debug/ReorderBuffer.hh"

#include <cassert>
#include <cstdint>

namespace gem5
{
ReorderBuffer::ReorderBuffer(const ReorderBufferParams &params):
    TickedObject(params), ROB_depth(params.ROB_depth), MDU_depth(params.MDU_depth)
{
    for(uint32_t i = 0; i < ROB_depth; i ++)
    {
        rob.push_back(new rob_entry(0,0,0));
    }
    tail = 0;
    head = 0;
    valid_element = 0;
    Occupied = false;

    for(uint32_t i = 0; i < 8; i ++)
    {
        for (uint32_t j = 0; j < MDU_depth; j ++)
        {
            memdep_table[i].push_back(memdep_entry());
        }
    }
}

ReorderBuffer::~ReorderBuffer()
{
}

void ReorderBuffer::set_matrixEnginePtr(MatrixEngine* _matrix_engine)
{
    matrix_engine = _matrix_engine;
}

void ReorderBuffer::startTicking()
{
    DPRINTF(ReorderBuffer, "Matrix Engine ReorderBuffer is start working!\n");
    Occupied = true;
    start();
}

void ReorderBuffer::stopTicking()
{
    DPRINTF(ReorderBuffer, "Matrix Engine ReorderBuffer is stop working!\n");
    Occupied = false;
    stop();
}

bool ReorderBuffer::isOccupied()
{
    return Occupied;
}

void ReorderBuffer::regStats()
{
    TickedObject::regStats();

    MatrixROBentryUsed
    .name(name() + ".Matrix_ROB_entry_use")
    .desc("Number of Matrix ROB entry used!");
    ROB_read
    .name(name() + ".ROBread")
    .desc("Number of ROB read");
    ROB_write
    .name(name() + ".ROBwrite")
    .desc("Number of ROB write");
}

void ReorderBuffer::evaluate()
{
    if((valid_element == 0)&&(std::all_of(mdvalid_element.begin(), mdvalid_element.end(),[](uint32_t v){ return v == 0; })))
    {
        stopTicking();
        return; //This return here is to stop left evaluate function body
    }

    /*Test ROB usage percentage*/
    if((double)valid_element > MatrixROBentryUsed.value())
    {
        MatrixROBentryUsed = valid_element; //主要是为了测试这个ROB能用到多深？
    }

    /*Commit the head ROB entry*/
    if((rob[head]->executed) && (rob[head]->valid))
    {
        DPRINTF(ReorderBuffer, "Commit the ROB head entry!\n");
        ROB_read++;
        if(rob[head]->old_dst_vld && rob[head]->old_dst != 99 && !matrix_engine->matrix_rename->checkLock(rob[head]->old_dst)){
            matrix_engine->matrix_rename->set_freeReg(rob[head]->old_dst); //conservative strategy to free the physical register
            std::cout << "free reg: " << rob[head]->old_dst << std::endl;
        }
        if(head == ROB_depth-1){
            head = 0;
        } else{
            head ++;
        }
        valid_element--;
    }

    for(uint8_t i = 0; i < 8; i ++)
    {
        /*Commit the head MD entry*/
        if(memdep_table[i][mdhead[i]].valid)
        {
            DPRINTF(ReorderBuffer, "Commit the MD head entry!\n");
            if(mdhead[i] == MDU_depth-1){
                mdhead[i] = 0;
            } else{
                mdhead[i] ++;
            }
            mdvalid_element[i] --;
        }
    }
}

bool ReorderBuffer::rob_full()
{
    return(valid_element == ROB_depth);
}

bool ReorderBuffer::md_full(uint8_t idx)
{
    return(mdvalid_element[idx] == MDU_depth);
}

uint32_t ReorderBuffer::set_md_entry(uint8_t idx)
{
    assert(mdvalid_element[idx] < MDU_depth);
    //insert to the tail entry
    uint32_t return_tail = mdtail[idx];
    memdep_table[idx][mdtail[idx]].valid = false;

    if(mdtail[idx] == MDU_depth-1){
        mdtail[idx] = 0;
    } else {
        mdtail[idx] ++;
    }

    mdvalid_element[idx] ++;
    return return_tail;
}

void ReorderBuffer::set_md_entry_valid(uint8_t idx, uint32_t entry_idx)
{
    memdep_table[idx][entry_idx].valid = true;
    memdep_table[idx][entry_idx].read_num = 0;
    if(mdhead[idx] == MDU_depth-1){
        mdhead[idx] = 0;
    } else {
        mdhead[idx] ++;
    }

    mdvalid_element[idx] --;
}

bool ReorderBuffer::rob_empty()
{
    return(valid_element == 0);
}

uint32_t ReorderBuffer::set_rob_entry(uint32_t logic_idx, uint32_t physic_idx, uint32_t old_dst, bool old_dst_vld)
{
    assert(valid_element < ROB_depth);
    //insert to the tail entry
    rob[tail]->logic_idx = logic_idx;
    rob[tail]->valid = true;
    rob[tail]->physic_idx = physic_idx;
    rob[tail]->executed = false;
    rob[tail]->old_dst = old_dst;
    rob[tail]->old_dst_vld = old_dst_vld;

    uint32_t return_tail = tail;
    if(tail == ROB_depth-1){
        tail = 0;
    } else {
        tail ++;
    }

    valid_element ++;
    ROB_write++;
    return return_tail;
}

void ReorderBuffer::set_rob_entry_executed(uint32_t idx)
{
    rob[idx]->executed = true;
}
} // namespace gem5
