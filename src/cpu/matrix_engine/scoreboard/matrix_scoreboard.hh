/*
 * @Author: superboy
 * @Date: 2024-06-19 19:58:37
 * @LastEditTime: 2024-07-09 17:13:16
 * @LastEditors: superboy
 * @Description: 
 * @FilePath: /gem5-rvm/src/cpu/matrix_engine/scoreboard/matrix_scoreboard.hh
 * 
 */
#ifndef __CPU_MATRIX_SCOREBOARD_HH__
#define __CPU_MATRIX_SCOREBOARD_HH__
#include "arch/riscv/insts/matrix_static_inst.hh"
#include <cstdint>
namespace gem5
{
class ScoreBoard_Entry
{
public:
    ScoreBoard_Entry(): renamed_src1(0), renamed_src2(0),
    renamed_src3(0), dst_lrf_num(0), dst_prf_num(0),
    rob_entry(0)
    {
        rs1_value = 0;
        rs2_value = 0;
        executed_num = 0;
        store_cnt = 0;
        load_cnt = 0;
        isIssue = false;
        isDone = false;
        isSent = false;
        isSendAndWait = false;
    }
    ~ScoreBoard_Entry(){}
    uint16_t get_renamed_src1() const { return renamed_src1; } //return physical register index
    void set_renamed_src1(uint16_t val)  { renamed_src1 = val; }

    uint16_t get_renamed_src2() const { return renamed_src2; }
    void set_renamed_src2(uint16_t val) { renamed_src2  = val; }

    uint16_t get_renamed_src3() const { return renamed_src3; }
    void set_renamed_src3(uint16_t val) { renamed_src3  = val; }

    uint16_t get_old_dst() const { return old_dst; }
    void set_old_dst(uint16_t val) { old_dst  = val; }

    uint16_t get_dst_lrf_num() const { return dst_lrf_num; }
    void set_dst_lrf_num(uint16_t val) { dst_lrf_num  = val; }

    uint16_t get_dst_prf_num() const { return dst_prf_num; }
    void set_dst_prf_num(uint16_t val) { dst_prf_num  = val; }

    uint16_t get_rob_entry() const { return rob_entry; }
    void set_rob_entry(uint16_t val) { rob_entry  = val; }

    uint64_t get_rs1_value() const { return rs1_value; }
    void set_rs1_value(uint64_t val) { rs1_value  = val; }

    uint64_t get_rs2_value() const { return rs2_value; }
    void set_rs2_value(uint64_t val) { rs2_value  = val; }

    RiscvISA::RiscvMatrixInst *get_MatrixStaticInst() const {
        return _minst;
    }
    void set_MatrixStaticInst(RiscvISA::RiscvMatrixInst *minst){
        _minst = minst;
    }

    void execute(){ 
        executed_num = executed_num + 1;
        printf("exe num = %d\n", executed_num);
    }
    uint8_t get_executed_num() const { return executed_num;}
    
    void store(){
        store_cnt = store_cnt + 1;
    }
    uint8_t get_store_cnt() const { return store_cnt;}

    void load(){
        load_cnt = load_cnt + 1;
    }
    uint8_t get_load_cnt() const { return load_cnt;}

    void set_cfg_size(uint8_t sizeN, uint8_t sizeM, uint16_t sizeK){
        _sizeN = sizeN;
        _sizeM = sizeM;
        _sizeK = sizeK;
    }
    
    uint8_t get_cfg_sizeN() const {return _sizeN;}
    uint8_t get_cfg_sizeM() const {return _sizeM;}
    uint16_t get_cfg_sizeK() const {return _sizeK;}

private:
    uint16_t renamed_src1;
    uint16_t renamed_src2;
    uint16_t renamed_src3;
    uint16_t old_dst;
    uint16_t dst_lrf_num;
    uint16_t dst_prf_num;
    uint16_t rob_entry;
    uint64_t rs1_value;
    uint64_t rs2_value;

    uint8_t executed_num; // for count executed number!
    uint8_t load_cnt;
    uint8_t store_cnt;

    //new struct here
    uint8_t _sizeN;
    uint8_t _sizeM;
    uint16_t _sizeK;
public:
    RiscvISA::RiscvMatrixInst *_minst;
    bool isIssue; //control signal of the matrix lane
    bool isDone; // control signal of the matrix lane
    bool isSent; // control signal of inst in memory_queue
    bool isSendAndWait; //以及开始去从memory获取或者向memory存数据了，但还没有真正的更新到寄存器中或者memory中。
    uint8_t lane_idx; //record which lane is executing this inst
};
}//namespace gem5

#endif