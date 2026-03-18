/*
 * @Author: superboy
 * @Date: 2025-10-04 11:25:46
 * @LastEditTime: 2025-10-22 17:32:05
 * @LastEditors: superboy
 * @Description: 
 * @FilePath: /SJTU-matrix-engine/src/cpu/matrix_engine/matrix_engine.cc
 * 
 */

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
MatrixEngine::MatrixEngine(const MatrixEngineParams &params) : SimObject(params), matrix_rename(params.matrix_rename), matrix_dispatcher(params.matrix_dispatcher), matrix_rob(params.matrix_rob), lane_num(params.lane_num), matrix_reg(params.matrix_reg), matrix_mmu(params.matrix_mmu), matrix_lanes(params.matrix_lanes), transpose_unit(params.transpose_unit), ew_unit(params.ew_unit), matrix_spm(params.matrix_spm)
{
    // for(uint8_t i = 0; i < lane_num; i++){ // we do this in the python script
    //     matrix_lanes.push_back(new Matrix)
    // }
    matrix_mmu->set_cpu_ptr(o3cpu);
    matrix_lanes.resize(params.lane_num);
    transpose_unit->set_matrixEnginePtr(this);
    // const std::vector<MatrixLane *> &matrix_lanes = params.matrix_lanes;
    for(int i=0; i < params.lane_num; i++){
        matrix_lanes[i]->set_matrixEnginePtr(this);
    }
    matrix_dispatcher->set_matrixEnginePtr(this);
    matrix_rename->set_matrixEnginePtr(this);
    matrix_rob->set_matrixEnginePtr(this);
    matrix_mmu->set_matrixEngine_ptr(this);
    ew_unit->set_matrixEnginePtr(this);
    matrix_spm->set_matrixEngine_ptr(this);
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

bool
MatrixEngine::MEreadSPM(Addr addr, uint32_t size, AccessMode mode,
        std::function<void(uint8_t* data, uint8_t size)> readCallback)
{
    // uint64_t req_id = (++uniqueReqId);
    // DPRINTF(MatrixEngine, "req_id = %llu is MEreadSPM with addr = %#llx, size = %u\n", 
    //     req_id, addr, size);

    // fixme: add check for addr and size alignment
    // Process *p = tc->getProcessPtr();
    // Addr page1 = p->pTable->pageAlign(addr);
    // Addr page2 = p->pTable->pageAlign(addr+size-1);
    // assert(page1 == page2);

    uint8_t *ndata = new uint8_t[size];
    memset(ndata, 'Z', size);
    MemCmd cmd = MemCmd::ReadReq;
    RequestPtr req = std::make_shared<Request>(addr, size, 0, 0); // Flags = 0?  RequestorID = 0?
    PacketPtr pkt = new SPMPacket(req, cmd, mode, readCallback);
    pkt->dataDynamic(ndata);
    DPRINTF(MatrixEngine, "MEreadSPM: addr = %#llx, size = %llu, cmd = %s, mode = %s\n",
        pkt->getAddr(), pkt->getSize(), pkt->cmdString(), AccessModeToString(mode));
    if(!matrix_spm->recvTimingReq(pkt)) {
        return false;
    }
    return true;
}

bool
MatrixEngine::MEwriteSPM(Addr addr, uint8_t* data, uint32_t size, AccessMode mode, std::function<void(void)> writeCallback)
{
    // uint64_t req_id = (++uniqueReqId);
    // DPRINTF(MatrixEngine, "req_id = %llu is MEwriteSPM with addr = %#llx, size = %u\n",
    //     req_id, addr, size);

    // fixme: add check for addr and size alignment
    // Process *p = tc->getProcessPtr();
    // Addr page1 = p->pTable->pageAlign(addr);
    // Addr page2 = p->pTable->pageAlign(addr+size-1);
    // assert(page1 == page2);

    uint8_t *ndata = new uint8_t[size];
    memcpy(ndata, data, size);
    MemCmd cmd = MemCmd::WriteReq;
    RequestPtr req = std::make_shared<Request>(addr, size, 0, 0); // Flags = 0?  RequestorID = 0?
    PacketPtr pkt = new SPMPacket(req, cmd, mode, writeCallback);
    pkt->dataDynamic(ndata);
    DPRINTF(MatrixEngine, "MEwriteSPM: addr = %#llx, size = %llu, cmd = %s, mode = %s\n",
        pkt->getAddr(), pkt->getSize(), pkt->cmdString(), AccessModeToString(mode));
    if(!matrix_spm->recvTimingReq(pkt)) {
        return false;
    }
    return true;
}

bool
MatrixEngine::Callback(PacketPtr pkt)
{
    // MatrixPacketPtr la_pkt = dynamic_cast<MatrixPacketPtr>(pkt);
    // assert(la_pkt != nullptr);
    DPRINTF(MatrixEngine, "callback is called in %llu\n", curTick());
    // DPRINTF(MatrixEngine, "rsp_pkt: reqid = %llu, cmd = %s\n",
    //     la_pkt->reqId, pkt->cmdString());
    uint8_t* data = pkt->getPtr<uint8_t>(); // 获取数据指针
    uint8_t size = pkt->getSize(); // 获取数据长度
    for(uint8_t i = 0; i < size; i++){
        // DPRINTF(MatrixEngine, "[wrp] callback: data[%u] = %u\n", i, data[i]);
    }
    SPMPacketPtr spm_pkt = dynamic_cast<SPMPacketPtr>(pkt);
    if (spm_pkt != nullptr) {
        if (spm_pkt->readCallback) {
            spm_pkt->readCallback(data, size);    
        } else if (spm_pkt->writeCallback) {
            spm_pkt->writeCallback();
        } else {
            DPRINTF(MatrixEngine, "Callback is not set\n");
        }       
    } else {
        DPRINTF(MatrixEngine, "Packet is not of type SPMPacketPtr in MatrixEngine::Callback\n");
    }
    delete pkt;
    return true;
}


}//namespace gem5