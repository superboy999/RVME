/*
 * @Author: superboy
 * @Date: 2024-06-19 19:58:37
 * @LastEditTime: 2025-10-26 15:31:03
 * @LastEditors: superboy
 * @Description: 
 * @FilePath: /SJTU-matrix-engine/src/cpu/matrix_engine/scoreboard/matrix_scoreboard.hh
 * 
 */
#ifndef __CPU_MATRIX_SCOREBOARD_HH__
#define __CPU_MATRIX_SCOREBOARD_HH__
#include "arch/riscv/insts/matrix_static_inst.hh"
#include "cpu/matrix_engine/spm/matrix_dma.hh"
#include <cstdint>
namespace gem5
{
class DmaReqState;
class ScoreBoard_Entry
{
public:
    ScoreBoard_Entry(): renamed_src1(0), renamed_src2(0),
    renamed_src3(0), dst_lrf_num(0), dst_prf_num(0),
    rob_entry(0)
    {
        rs1_value = 0;
        rs2_value = 0;
        rd_value = 0;
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

    uint64_t get_rd_value() const { return rd_value; }
    void set_rd_value(uint64_t val) { rd_value  = val; }

    uint64_t get_rs2_value() const { return rs2_value; }
    void set_rs2_value(uint64_t val) { rs2_value  = val; }

    uint64_t get_uimm5_value() const {
        return uimm5;
    }
    void set_uimm5_value(uint64_t val) {
        uimm5 = val;
    }

    void setDoneCallback(const std::function<void(uint64_t)>& cb) {
        doneCallback = cb;
    }
    void callDoneCallback(uint64_t val) {
        if(doneCallback)
            doneCallback(val);
    }

    RiscvISA::RiscvMatrixInst *get_MatrixStaticInst() const {
        return _minst;
    }
    void set_MatrixStaticInst(RiscvISA::RiscvMatrixInst *minst){
        _minst = minst;
    }

    void execute(){ 
        executed_num = executed_num + 1;
        // printf("exe num = %d\n", executed_num);
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

    uint8_t ms1 = 0;
    uint8_t ms2 = 0;
    uint8_t ms3 = 0;
    uint8_t md = 0;

private:
    uint16_t renamed_src1;
    uint16_t renamed_src2;
    uint16_t renamed_src3;
    uint16_t old_dst = 99; //不专门设置old_dst的话，默认为99，不需要在rob中set old dst free.
    uint16_t dst_lrf_num;
    uint16_t dst_prf_num;
    uint16_t rob_entry;
    uint64_t rs1_value;
    uint64_t rd_value;
    uint64_t rs2_value;
    uint64_t uimm5;
    
    std::function<void(uint64_t)> doneCallback;

    uint8_t executed_num; // for count executed number!
    uint8_t load_cnt;
    uint8_t store_cnt;

    //new struct here
    uint8_t _sizeN;
    uint8_t _sizeM;
    uint16_t _sizeK;

    
public:
    RiscvISA::RiscvMatrixInst *_minst;
    bool canIssue = false; //control signal of the matrix lane
    bool isIssue = false; //control signal of the matrix lane
    bool isDone = false; // control signal of the matrix lane
    bool isSent = false; // control signal of inst in memory_queue
    bool isSendAndWait = false; //以及开始去从memory获取或者向memory存数据了，但还没有真正的更新到寄存器中或者memory中。
    bool istranspose;
    uint8_t matrix_type = 0; // record load / store matrix type, 0 for a, 1 for b, 2 for c
    uint32_t load_data_size = 0;
    uint8_t lane_idx = 0; //record which lane is executing this inst
    uint8_t ewu_idx = 0; //record which ewu is executing this inst

    uint8_t src1_read_idx = 0;
    uint8_t src2_read_idx = 0;
    uint8_t src3_read_idx = 0;
    uint8_t dst_read_idx = 0;

    uint32_t dst_memdep_idx = 0; // record the tail of mdtable when setting the md entry, and RAW is solved when memdep_idx equals to the mdtable head.
    uint32_t src1_memdep_idx = 0;
    uint32_t src2_memdep_idx = 0;
    uint32_t src3_memdep_idx = 0;
    uint64_t spm_reqId = 0;
private:
// SPM config datas
uint64_t addr = 0;
uint32_t stride = 0;
uint32_t row_size = 0; //一行多少bytes
uint8_t mode = 0;
uint8_t setid = 0;
uint8_t rows = 0;

uint64_t mem_addr = 0;
uint64_t spm_addr = 0;
uint8_t spm_uimm5 = 0; // 

public:
    uint64_t get_mem_addr() const { return mem_addr; }
    void set_mem_addr(uint64_t val) { mem_addr = val; }

    uint64_t get_spm_addr() const { return spm_addr; }
    void set_spm_addr(uint64_t val) { spm_addr = val; }

    uint8_t get_spm_uimm5() const { return spm_uimm5; }
    void set_spm_uimm5(uint8_t val) { spm_uimm5 = val; }


    uint8_t get_mode() const { return mode; }
    void set_mode(uint8_t val) { mode = val; }

    uint8_t get_setid() const { return setid; }
    void set_setid(uint8_t val) { setid = val; }

    uint8_t get_rows() const { return rows; }
    void set_rows(uint8_t val) { rows = val; }

    uint64_t get_addr() const { return addr; }
    void set_addr(uint64_t val) { addr = val; }

    uint64_t get_data_size() const { return load_data_size; }
    void set_data_size(uint64_t val) { load_data_size = val; }

    uint32_t get_stride() const { return stride; }
    void set_stride(uint32_t val) { stride = val; }

    uint8_t get_row_size() const { return row_size; }
    void set_row_size(uint32_t val) { row_size = val; }

    std::shared_ptr<DmaReqState> spmReq;
    void setDmaReqState(std::shared_ptr<DmaReqState> _spmReq){
        spmReq = _spmReq;
    }
};
}//namespace gem5

#endif