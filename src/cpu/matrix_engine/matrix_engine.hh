// ==========================================
// Author: cwq
// Last Date: 2023/12/21
// ==========================================
#ifndef __CPU_MATRIX_ENGINE_HH__
#define __CPU_MATRIX_ENGINE_HH__

#include <cassert>
#include <cstdint>
#include <vector>

#include "sim/sim_object.hh"
#include "params/MatrixEngine.hh"
#include "arch/riscv/insts/matrix_static_inst.hh"
#include "cpu/matrix_engine/matrix_lane/Matrix_Lane.hh"
#include "cpu/matrix_engine/rename/matrix_rename.hh"
#include "cpu/matrix_engine/reorder_buffer/reorder_buffer.hh"
#include "cpu/matrix_engine/dispatcher/matrix_dispatcher.hh"
#include "cpu/matrix_engine/matrix_registerfile/matrix_reg.hh"
#include "cpu/matrix_engine/mmu/matrix_mmu.hh"
#include "cpu/o3/cpu.hh"
namespace gem5
{
struct MatrixEngineParams;
class MatrixEngine : public SimObject
{
public:
    MatrixEngine(const MatrixEngineParams &params);
    ~MatrixEngine(){};

    // void configMatrix(RiscvISA::RiscvMatrixInst &minst, uint64_t src1);
    void cfgSizeN(uint8_t sizen);
    void cfgSizeM(uint8_t sizem);
    void cfgSizeK(uint8_t sizek);
    uint8_t get_sizeM();
    uint8_t get_sizeN();
    uint16_t get_sizeK();
    bool isOccupied();
    void set_cpu_ptr(gem5::o3::CPU* _o3cpu);

private:
    // cwq I hope this value will be saved in two space, one is here and the other one will be inside the Matrix RegisterFile.
    // But they should be synchronize at same time when the configure instruction come into engine.
    uint8_t matrix_sizeM; // eight logic register are all the same controlled by one Msize Register
    uint8_t matrix_sizeN; 
    uint16_t matrix_sizeK;

public:
    std::vector<MatrixLane* > matrix_lanes; //Number will be assigned in the python script
    MatrixRename *matrix_rename;
    ReorderBuffer *matrix_rob;
    MatrixDispatcher *matrix_dispatcher;
    MatrixRF *matrix_reg;
    MatrixMMU *matrix_mmu;
    //python configuration
    uint8_t lane_num;
    gem5::o3::CPU* o3cpu;
};

} //namespace gem5

#endif