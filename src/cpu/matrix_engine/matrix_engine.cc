// ==========================================
// Author: cwq
// Last Date: 2023/12/21
// ==========================================

#include <algorithm>
#include <functional>
#include <numeric>
#include <string>
#include <vector>
#include <cassert>
#include <cstdint>

#include "debug/MatrixEngine.hh"
#include "cpu/matrix_engine/matrix_engine.hh"

namespace gem5
{
MatrixEngine::MatrixEngine(const MatrixEngineParams &params) : SimObject(params), matrix_rename(params.matrix_rename), matrix_dispatcher(params.matrix_dispatcher), matrix_rob(params.matrix_rob), lane_num(params.lane_num), matrix_reg(params.matrix_reg), matrix_mmu(params.matrix_mmu), matrix_lanes(params.matrix_lanes)
{
    // for(uint8_t i = 0; i < lane_num; i++){ // we do this in the python script
    //     matrix_lanes.push_back(new Matrix)
    // }
    matrix_mmu->set_cpu_ptr(o3cpu);
    matrix_lanes.resize(params.lane_num);
    // const std::vector<MatrixLane *> &matrix_lanes = params.matrix_lanes;
    for(int i=0; i < params.lane_num; i++){
        matrix_lanes[i]->set_matrixEnginePtr(this);
    }
    matrix_dispatcher->set_matrixEnginePtr(this);
    matrix_rob->set_matrixEnginePtr(this);
    matrix_mmu->set_matrixEngine_ptr(this);
}

void MatrixEngine::cfgSizeN(uint8_t sizen)
{
    matrix_sizeN = sizen;
}
void MatrixEngine::cfgSizeM(uint8_t sizem)
{
    matrix_sizeM = sizem;
}
void MatrixEngine::cfgSizeK(uint8_t sizek)
{
    matrix_sizeK = sizek;
}

uint8_t MatrixEngine::get_sizeM()
{
    return matrix_sizeM;
}

uint8_t MatrixEngine::get_sizeN()
{
    return matrix_sizeN;
}

uint16_t MatrixEngine::get_sizeK()
{
    return matrix_sizeK;
}

bool MatrixEngine::isOccupied()
{
    //FIXME: cwq maybe
    bool busy = false;
    for(uint16_t i = 0; i < lane_num; i++){
        busy = matrix_lanes[i]->isOccupied();
    }
    return(busy);
}

void MatrixEngine::set_cpu_ptr(gem5::o3::CPU* _o3cpu)
{
    o3cpu = _o3cpu;
    matrix_mmu->set_cpu_ptr(o3cpu);
}

}//namespace gem5