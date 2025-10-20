/*
 * @Author: superboy
 * @Date: 2025-10-04 21:12:45
 * @LastEditTime: 2025-10-19 02:28:52
 * @LastEditors: superboy
 * @Description: Has same size with Lane number, each lane has one EWU for accumulation, maybe for elementwise operation is redundant.
 * @FilePath: /gem5-rvm/src/cpu/matrix_engine/elementwise_unit/ew_unit.cc
 * 
 */

#include "cpu/matrix_engine/elementwise_unit/ew_unit.hh"
#include "debug/ElementwiseUnit.hh"
#include "cpu/matrix_engine/matrix_registerfile/matrix_reg.hh"
#include "cpu/matrix_engine/scoreboard/matrix_scoreboard.hh"
#include <cassert>
#include <cstdint>

namespace gem5
{
ElementwiseUnit::ElementwiseUnit(const ElementwiseUnitParams &params):
    TickedObject(params), parallel_ewu(params.parallel_ewu)
{
    busy.resize(parallel_ewu, false);
    done.resize(parallel_ewu, false);
    isSigned.resize(parallel_ewu, false);
    exe_num.resize(parallel_ewu, 0);
    sizem.resize(parallel_ewu, 0);
    delay_cycle.resize(parallel_ewu, 0);
    src1.resize(parallel_ewu, 0);
    src2.resize(parallel_ewu, 0);
    dest.resize(parallel_ewu, 0);
    accReq.resize(parallel_ewu, false);
    operaReq.resize(parallel_ewu, false);
    isExe.resize(parallel_ewu, false);
    cur_ew_op.resize(parallel_ewu);
}

ElementwiseUnit::~ElementwiseUnit(){}

void ElementwiseUnit::set_matrixEnginePtr(MatrixEngine* _matrix_engine)
{
    matrix_engine = _matrix_engine;
}

void ElementwiseUnit::startTicking()
{
    // assert(!busy && "Elementwise Unit is busy!");
    // busy = true;
    start();
}

void ElementwiseUnit::stopTicking()
{
    // assert(busy && "Elementwise Unit is not busy!");
    // busy = false;
    stop();
}

bool ElementwiseUnit::isOccupied(uint8_t idx)
{
    return busy[idx];
}

bool ElementwiseUnit::isIdle()
{
    for(int i = 0; i < parallel_ewu; i++)
    {
        if(busy[i])
            return false;
    }
    return true;
}

void ElementwiseUnit::regStats()
{
    TickedObject::regStats();
    ew_op_accepted
        .name(name() + ".ew_op_accepted")
        .desc("Number of elementwise operations accepted");
    acc_executed
        .name(name() + ".acc_executed")
        .desc("Number of accumulate operations executed");
}

void ElementwiseUnit::acc_req(uint8_t _dest, uint8_t _sizem, uint8_t _idx, bool _isSigned) // This is special because this will accept value directly from zbuffer(one OPA) 
{
    sizem[_idx] = _sizem;
    busy[_idx] = true;
    done[_idx] = false;
    isSigned[_idx] = _isSigned;
    delay_cycle[_idx] = 1; //accumulate delay is 1 cycle
    DPRINTF(ElementwiseUnit, "EWU[%d] accumulate request accepted: dest=%d, sizem=%d\n", _idx, _dest, _sizem);
    accReq[_idx] = true;
    dest[_idx] = _dest;
    acc_executed++;
    // startTicking();
}

void ElementwiseUnit::recv_opera(ScoreBoard_Entry& matrix_sbe, uint8_t _idx, bool _isSigned) // This is special because this will accept value directly from ArithQueue(one OPA)
{
    if(matrix_sbe._minst->ismadd())
        cur_ew_op[_idx] = MatrixInstClass::EW_OP::MADD;
    else if(matrix_sbe._minst->ismsub())
        cur_ew_op[_idx] = MatrixInstClass::EW_OP::MSUB;
    else if(matrix_sbe._minst->ismmul())
        cur_ew_op[_idx] = MatrixInstClass::EW_OP::MMUL;
    else if(matrix_sbe._minst->ismshiftr())
        cur_ew_op[_idx] = MatrixInstClass::EW_OP::MSHITFTR;
    else
        panic("Unsupported EW_OP!");
    // cur_ew_op[_idx] = ew_op;
    sizem[_idx] = matrix_sbe.get_cfg_sizeM();
    busy[_idx] = true;
    done[_idx] = false;
    isSigned[_idx] = _isSigned;
    delay_cycle[_idx] = OP_delay[cur_ew_op[_idx]];
    src1[_idx] = matrix_sbe.get_renamed_src1();
    src2[_idx] = matrix_sbe.get_renamed_src2();
    dest[_idx] = matrix_sbe.get_dst_prf_num();
    operaReq[_idx] = true;
    DPRINTF(ElementwiseUnit, "EWU[%d] operation accepted: ew_op=%d, src1=%d, src2=%d, dest=%d\n", _idx, cur_ew_op[_idx], src1[_idx], src2[_idx], dest[_idx]);
    ew_op_accepted++; 
    // startTicking();
}

std::array<int32_t, 8> ElementwiseUnit::elementwise_operation_s(std::array<int32_t, 8> A, std::array<int32_t, 8> B, MatrixInstClass::EW_OP ew_op)
{
    std::array<int32_t, 8> result;
    switch(ew_op)
    {
        case MatrixInstClass::EW_OP::MADD:
            for(int i = 0; i < 8; i++)
                result[i] = A[i] + B[i];
            break;
        case MatrixInstClass::EW_OP::MSUB:
            for(int i = 0; i < 8; i++)
                result[i] = A[i] - B[i];
            break;
        case MatrixInstClass::EW_OP::MMUL:
            for(int i = 0; i < 8; i++)
                result[i] = A[i] * B[i];
            break;
        case MatrixInstClass::EW_OP::MMAX:
            for(int i = 0; i < 8; i++)
                result[i] = std::max(A[i], B[i]);
            break;
        case MatrixInstClass::EW_OP::MMIN:  
            for(int i = 0; i < 8; i++)
                result[i] = std::min(A[i], B[i]);
            break;
        case MatrixInstClass::EW_OP::MSHITFTR:
            for(int i = 0; i < 8; i++)
                result[i] = A[i] >> B[i]; //logical shift right
            break;
        case MatrixInstClass::EW_OP::MSHITFTL:
            for(int i = 0; i < 8; i++)
                result[i] = A[i] << B[i]; //logical shift left
            break;
        default:
            panic("Unsupported EW_OP!");
    }
    return result;
}

std::array<uint32_t, 8> ElementwiseUnit::elementwise_operation_u(std::array<uint32_t, 8> A, std::array<uint32_t, 8> B, MatrixInstClass::EW_OP ew_op)
{
    std::array<uint32_t, 8> result;
    switch(ew_op)
    {
        case MatrixInstClass::EW_OP::MADD:
            for(int i = 0; i < 8; i++)
                result[i] = A[i] + B[i];
            break;
        case MatrixInstClass::EW_OP::MSUB:
            for(int i = 0; i < 8; i++)
                result[i] = A[i] - B[i];
            break;
        case MatrixInstClass::EW_OP::MMUL:
            for(int i = 0; i < 8; i++)
                result[i] = A[i] * B[i];
            break;
        case MatrixInstClass::EW_OP::MMAX:
            for(int i = 0; i < 8; i++)
                result[i] = std::max(A[i], B[i]);
            break;
        case MatrixInstClass::EW_OP::MMIN:  
            for(int i = 0; i < 8; i++)
                result[i] = std::min(A[i], B[i]);
            break;
        case MatrixInstClass::EW_OP::MSHITFTR:
            for(int i = 0; i < 8; i++)
                result[i] = A[i] >> B[i]; //logical shift right
            break;
        case MatrixInstClass::EW_OP::MSHITFTL:
            for(int i = 0; i < 8; i++)
                result[i] = A[i] << B[i]; //logical shift left
            break;
        default:
            panic("Unsupported EW_OP!");
    }
    return result;
}

void ElementwiseUnit::evaluate()
{
    for(int i = 0; i < parallel_ewu; i++)
    {
        if(busy[i])
        {
            if(delay_cycle[i] > 1)
            {
                delay_cycle[i]--;
            }
            else //execute the operation
            {   
                bool success = false;
                
                if(operaReq[i])
                {
                    bool ocpy_mrf = matrix_engine->matrix_reg->occupy_rdport(src1[i], 0, exe_num[i])&&matrix_engine->matrix_reg->occupy_rdport(src2[i], 0, exe_num[i])&&matrix_engine->matrix_reg->occupy_wtport(dest[i], 0, exe_num[i]);
                    std::array<uint32_t, 8> A_u{}, B_u{};
                    std::array<int32_t, 8> A_s{}, B_s{};
                    if(ocpy_mrf || isExe[i])
                    {
                        isExe[i] = true;
                        A_u = [this, i](int size) {
                            std::array<uint32_t, 8> result{};
                            for(int j = 0; j < size; j++)
                            {
                                result[j] = matrix_engine->matrix_reg->rdreg_int32(this->src1[i], j / 2, this->exe_num[i], j%2);
                            }
                            return result;
                        }(8);
                        A_s = [this, i](int size) {
                            std::array<int32_t, 8> result{};
                            for(int j = 0; j < size; j++)
                            {
                                result[j] = matrix_engine->matrix_reg->rdreg_int32(this->src1[i], j / 2, this->exe_num[i], j%2);
                            }
                            return result;
                        }(8);
                        B_u = [this, i](int size) {
                            std::array<uint32_t, 8> result{};
                            for(int j = 0; j < size; j++)
                            {
                                result[j] = matrix_engine->matrix_reg->rdreg_int32(this->src2[i], j / 2, this->exe_num[i], j%2);
                            }
                            return result;
                        }(8);
                        B_s = [this, i](int size) {
                            std::array<int32_t, 8> result{};
                            for(int j = 0; j < size; j++)
                            {
                                result[j] = matrix_engine->matrix_reg->rdreg_int32(this->src2[i], j / 2, this->exe_num[i], j%2);
                            }
                            return result;
                        }(8);
                        std::array<int32_t, 8> RESULT_s{};
                        std::array<uint32_t, 8> RESULT_u{};
                        if(isSigned[i]){
                            RESULT_s = elementwise_operation_s(A_s, B_s, cur_ew_op[i]);
                        } else {
                            RESULT_u = elementwise_operation_u(A_u, B_u, cur_ew_op[i]);
                        }
                        for(uint8_t j = 0; j < 8; j++)
                        {
                            matrix_engine->matrix_reg->wtreg_int32(dest[i], j / 2, exe_num[i], j%2, isSigned[i] ? RESULT_s[j] : RESULT_u[j]);
                        }
                        DPRINTF(ElementwiseUnit, "EWU[%d] operation executed: ew_op=%d, src1=%d, src2=%d, dest=%d\n", i, cur_ew_op[i], src1[i], src2[i], dest[i]);
                        success = true;
                        matrix_engine->matrix_reg->rls_rdport(src1[i], 0, exe_num[i]);
                        matrix_engine->matrix_reg->rls_rdport(src2[i], 0, exe_num[i]);
                        matrix_engine->matrix_reg->rls_wrport(dest[i], 0, exe_num[i]);
                    } else {
                        DPRINTF(ElementwiseUnit, "EWU[%d] operation failed and is waiting due to no read/write port: ew_op=%d, src1=%d, src2=%d, dest=%d\n", i, cur_ew_op[i], src1[i], src2[i], dest[i]);
                    }
                }
                else if(accReq[i])
                {
                    std::array<uint32_t, 8> C{}, AB{}, D{};
                    // if(isExe[i]) // 这里不需要ocy_mrf, 因为zbuffer做了occupy_wtport
                    // {
                    isExe[i] = true;
                    AB = matrix_engine->matrix_lanes[i]->zbuffer->send2EWU_u(); //FIXME: 要把这个输出和dest的相加
                    DPRINTF(ElementwiseUnit, "EWU[%d] accumulate received from MatrixRF: dest=%d\n", i, dest[i]);
                    DPRINTF(ElementwiseUnit, "EWU[%d] receive from ZBuffer: %d, %d, %d, %d, %d, %d, %d, %d\n", i, AB[0], AB[1], AB[2], AB[3], AB[4], AB[5], AB[6], AB[7]);
                    C = [i, this](int size) {
                        std::array<uint32_t, 8> result{};
                        for(int j = 0; j < size; j++)
                        {
                            result[j] = matrix_engine->matrix_reg->rdreg_int32(this->dest[i], j / 2, this->exe_num[i], j%2);
                        }
                        return result;
                    }(8);
                    DPRINTF(ElementwiseUnit, "EWU[%d] accumulate read from MatrixRF: dest=%d, %d, %d, %d, %d, %d, %d, %d\n", i, dest[i], C[0], C[1], C[2], C[3], C[4], C[5], C[6], C[7]);
                    //D = AB + C;
                    for (uint8_t j = 0; j < 8; j++) {
                        D[j] = AB[j] + C[j];
                    }
                    DPRINTF(ElementwiseUnit, "EWU[%d] accumulate result: , %d, %d, %d, %d, %d, %d, %d, %d\n", i, D[0], D[1], D[2], D[3], D[4], D[5], D[6], D[7]);
                    for(uint8_t j = 0; j < 8; j++)
                    {
                        matrix_engine->matrix_reg->wtreg_int32(dest[i], j / 2, exe_num[i], j%2, D[j]);
                    }
                    // acc_req[i] = false;
                    success = true;
                    matrix_engine->matrix_reg->rls_rdport(src1[i], 0, exe_num[i]);
                    matrix_engine->matrix_reg->rls_wrport(dest[i], 0, exe_num[i]);
                    // } else {
                    //     DPRINTF(ElementwiseUnit, "EWU[%d] accumulate failed and is waiting due to no read port: dest=%d\n", i, dest[i]);
                    // }
                }
                if(success)
                {
                    // busy[i] = false;
                    exe_num[i]++;
                    if(exe_num[i] == sizem[i])
                    {
                        busy[i] = false;
                        exe_num[i] = 0;
                        sizem[i] = 0;
                        isExe[i] = false;
                        done[i] = true;
                        // matrix_engine->matrix_reg->rls_rdport();
                        // matrix_engine->matrix_reg->rls_wrport();
                        if(operaReq[i])
                            operaReq[i] = false;
                            // matrix_engine->matrix_reg->rls_rdport(src1[i], 0, exe_num[i]);
                            // matrix_engine->matrix_reg->rls_rdport(src2[i], 0, exe_num[i]);
                            // matrix_engine->matrix_reg->rls_wrport(dest[i], 0, exe_num[i]);
                        if(accReq[i])
                            accReq[i] = false;
                            // matrix_engine->matrix_reg->rls_wrport(); // has been released in zbuffer
                        DPRINTF(ElementwiseUnit, "EWU[%d] all operations done\n", i);
                        
                    }
                }
            }
        }
    }
    bool all_idle = true;
    for(int i = 0; i < parallel_ewu; i++)
    {
        if(busy[i])
        {
            all_idle = false;
            break;
        }
    }
    // if(all_idle)
    //     stopTicking();
}


} //namespace gem5