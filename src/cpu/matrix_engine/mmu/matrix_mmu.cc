#include "cpu/matrix_engine/mmu/matrix_mmu.hh"
#include "debug/MatrixMMU.hh"
#include "sim/system.hh"

#include <cassert>
#include <cstdint>
#include <string>
#include <memory>
#include <functional>

namespace gem5
{
MatrixMMU::MatrixMMU(const MatrixMMUParams &params) :ClockedObject(params), uniqueReqId(0), num_matrix_mem_ports(params.num_matrix_mem_ports), w_channel(params.w_channel), r_channel(params.r_channel), pending_depth(params.pending_depth)
{
    for(uint8_t i = 0; i < params.num_matrix_mem_ports; i++)
    {
        MatrixCacheRequestorIds.push_back(params.system->getRequestorId(this, name() + ".matrix_mem" + std::to_string(i)));
        matrix_mem_ports.push_back(MatrixMemPort(name() + ".matrix_mem_ports", this, i));
        // std::deque<MatrixPacketPtr> initialDeque;
        // matrix_PendingReqQs.push_back(initialDeque);
    }
    matrix_PendingReqQs.resize(params.num_matrix_mem_ports);
}

MatrixMMU::~MatrixMMU()
{
}

void MatrixMMU::regStats()
{
    ClockedObject::regStats();
    // FIXME:
}

// void MatrixMMU::evaluate()
// {
//     FIXME:
// }

// MatrixMMU::MatrixMemPort::MatrixMemPort(const std::string& name, MatrixMMU* owner, uint8_t port_id) :
// RequestPort(name, owner), owner(owner), port_id(port_id)
// {
//     // for(uint8_t i = 0; i < owner->num_matrix_mem_ports; i++){
//     //     for(uint8_t j=0; j < 2; j++)
//     //     {
//     //         laCachePktQs[i][j].push_back(PacketPtr());
//     //     }
//     // }
//     laCachePktQs.resize(owner->num_matrix_mem_ports);
//     for (auto &port_queue : laCachePktQs) {
//         port_queue.resize(2);  // 假设每个端口有两个通道
//     }   

// }

MatrixMMU::MatrixMemPort::MatrixMemPort(const std::string& name, MatrixMMU* owner, uint8_t port_id) :
    RequestPort(name, owner), owner(owner), port_id(port_id)
{
    // 初始化laCachePktQs，确保每个端口有两个通道的缓存队列
    laCachePktQs.resize(owner->num_matrix_mem_ports);
    for (auto &port_queue : laCachePktQs) {
        port_queue.resize(2);  // 假设每个端口有两个通道
    }
}

MatrixMMU::MatrixMemPort::~MatrixMemPort()
{}

MatrixMMU::MatrixMemPort::MMU_Translation::MMU_Translation(MatrixMMU* owner) :
    event(this, true), owner(owner)
{}

MatrixMMU::MatrixMemPort::MMU_Translation::~MMU_Translation()
{}

void MatrixMMU::MatrixMemPort::MMU_Translation::markDelayed()
{
    panic("MMU Translation::markDelayed not implemented");
}

void MatrixMMU::MatrixMemPort::MMU_Translation::finish(const Fault &_fault,const RequestPtr &_req, ThreadContext *_tc, BaseMMU::Mode _mode)
{
    fault = _fault;
}

void MatrixMMU::MatrixMemPort::MMU_Translation::finish(const Fault _fault, uint64_t latency)
{
    fault = _fault;
}

std::string MatrixMMU::MatrixMemPort::MMU_Translation::name()
{
    return "MMU_Translation";
}

void MatrixMMU::MatrixMemPort::MMU_Translation::translated()
{
}

//If is read operation, data ptr will be assigned nullptr; If is write operation, data ptr will be assigned write data;
bool MatrixMMU::MatrixMemPort::startTranslation(Addr addr, uint8_t* data, uint32_t size, BaseMMU::Mode mode, ThreadContext *tc, uint32_t req_id, uint8_t channel, uint8_t port_id)
{
    //To ensure that a single access does not exceed the range of one page table.
    Process *p = tc->getProcessPtr();
    Addr page1 = p->pTable->pageAlign(addr);
    Addr page2 = p->pTable->pageAlign(addr+size-1);
    assert(page1 == page2);

    uint8_t *ndata = new uint8_t[size];
    if(data != nullptr){
        assert (mode == BaseMMU::Write);
        memcpy(ndata, data, size);
    } else {
        //assert (mode == BaseMMU::Read);
        memset(ndata, 'Z', size);
    }
    // build A new 'req'
    MemCmd cmd = (mode==BaseMMU::Write) ? MemCmd::WriteReq : MemCmd::ReadReq;
    const Addr pc = tc->pcState().instAddr();
    RequestPtr req = std::make_shared<Request>(addr, size, 0, owner->MatrixCacheRequestorIds[0], pc, tc->contextId());

    // FIXME: dont know what to do with it
    // o3::CPU *cpu = tc->getCpuPtr();
    req->taskId(owner->o3cpu->taskId());

    MMU_Translation* translation = new MMU_Translation(owner);
    // translation->translateTiming(req, tc, translation, mode);
    owner->o3cpu->mmu->dtb->translateTiming(req, tc, translation, mode);

    if(translation->fault == NoFault){
        PacketPtr pkt = new MatrixPacket(req, cmd, req_id, channel);
        pkt->dataDynamic(ndata);
        if (!sendTimingReq(pkt)) {
            laCachePktQs[port_id][channel].push_back(pkt);
        }
        delete translation;
        DPRINTF(MatrixMMU, "translation success!!\n");
        return true;
    } else {
        translation->fault->invoke(tc, NULL);
        DPRINTF(MatrixMMU, "translation fail!!\n");
        delete translation;
        return false;
    }
}

bool MatrixMMU::MatrixMemPort::sendTimingReadReq(Addr addr, uint32_t size, ThreadContext *tc, uint32_t req_id, uint8_t channel, uint8_t port_id)
{
    if (laCachePktQs[port_id][channel].size() >= owner->pending_depth) {
        // 队列已满，无法发送请求
        return false;
    }
    return startTranslation(addr, static_cast<uint8_t*>(nullptr), size, BaseMMU::Read, tc, req_id, channel, port_id);
}

bool MatrixMMU::MatrixMemPort::sendTimingWriteReq(Addr addr, uint8_t *data, uint32_t size, ThreadContext *tc, uint32_t req_id, uint8_t channel, uint8_t port_id)
{
    if (laCachePktQs[port_id][channel].size() >= owner->pending_depth) {
        // 队列已满，无法发送请求
        return false;
    }
    return startTranslation(addr, data, size, BaseMMU::Write, tc, req_id, channel, port_id);
}

bool MatrixMMU::MatrixMemPort::recvTimingResp(PacketPtr pkt)
{
    MatrixPacketPtr la_pkt = dynamic_cast<MatrixPacketPtr>(pkt);
    owner->recvTimingResp(la_pkt);
    return true;
}



// void MatrixMMU::MatrixMemPort::recvReqRetry()
// {
//     for (auto& laCachePktQ : laCachePktQs) {
//         for (auto& pktQ : laCachePktQ){
//             while (pktQ.size() && sendTimingReq(pktQ.front())) {
//                 pktQ.pop_front();
//             }
//         }

//     }
// }

void MatrixMMU::MatrixMemPort::recvReqRetry()
{
    for (auto& laCachePktQ : laCachePktQs) {
        for (auto& pktQ : laCachePktQ) {
            while (!pktQ.empty() && sendTimingReq(pktQ.front())) {
                pktQ.pop_front();
            }
        }
    }
}

// void MatrixMMU::recvTimingResp(MatrixPacketPtr pkt)
// {
//     assert(matrix_PendingReqQs.size());
//     bool match = false;
//     for(uint8_t i = 0; i < num_matrix_mem_ports; i++){
//         for (auto it = matrix_PendingReqQs[i].begin(); it != matrix_PendingReqQs[i].end(); ++it){
//             if (pkt->reqId == (*it)->reqId) {
//                 pkt->readCallback = (*it)->readCallback;
//                 pkt->writeCallback = (*it)->writeCallback;
//                 *it = pkt;  // 使用迭代器修改容器中的指针
//                 assert(pkt->readCallback);
//                 (*it)->finish = true;
//                 match = true;
//                 break;
//             }
//         }
//         if (match) break;
//     }

//     assert(match);

//     for(auto& matrix_PendingReqQ: matrix_PendingReqQs){
//         while(!matrix_PendingReqQ.empty() && matrix_PendingReqQ.front()->finish){
//             MatrixPacketPtr mpkt = matrix_PendingReqQ.front();
//             matrix_PendingReqQ.pop_front();
//             if(mpkt->cmd == MemCmd::WriteReq){
//                 if(mpkt->writeCallback){//ensure callback function is set.
//                     mpkt->writeCallback();
//                     // FIXME: //ADD specific function
//                 } else {
//                     panic("writeCallback is not set");
//                 }
//             } else{
//                 if(mpkt->readCallback){//ensure callback function is set.
//                     mpkt->readCallback(mpkt->getPtr<uint8_t>(), mpkt->getSize());
//                     // assert(mpkt->getSize() == 32);
//                     // assert(mpkt->getPtr<uint8_t>()[31] != 0);
//                 }
//                 else{
//                     panic("readCallback is not set");
//                 }
//             }
//             delete mpkt;
//         }
//     }
// }

void MatrixMMU::recvTimingResp(MatrixPacketPtr pkt)
{
    assert(matrix_PendingReqQs.size());
    bool match = false;

    // 遍历所有的端口和其对应的请求队列，寻找匹配的请求包
    for (uint8_t i = 0; i < num_matrix_mem_ports; i++) {
        for (auto it = matrix_PendingReqQs[i].begin(); it != matrix_PendingReqQs[i].end(); ++it) {
            if (pkt->reqId == (*it)->reqId) {
                pkt->readCallback = (*it)->readCallback;
                pkt->writeCallback = (*it)->writeCallback;
                pkt->dst_reg = (*it)->dst_reg; //fix this bug
                pkt->cmd = (*it)->cmd;
                *it = pkt;  // 使用迭代器修改容器中的指针
                assert(pkt->readCallback || pkt->writeCallback);
                (*it)->finish = true;
                match = true;
                break;
            }
        }
        if (match) break;
    }

    assert(match); // 如果没有找到匹配项则断言失败

    // 处理已完成的请求包
    for (auto& matrix_PendingReqQ : matrix_PendingReqQs) {
        while (!matrix_PendingReqQ.empty() && matrix_PendingReqQ.front()->finish) {
            MatrixPacketPtr mpkt = matrix_PendingReqQ.front();
            matrix_PendingReqQ.pop_front();
            if (mpkt->cmd == MemCmd::WriteReq) {
                if (mpkt->writeCallback) { //just for debug, test if mpkt is correctly assigned with writeback function
                    mpkt->writeCallback();
                } else {
                    panic("writeCallback is not set");
                }
            } else {
                if(!matrix_engine->matrix_reg->try_occupy(mpkt->dst_reg)){
                    break;
                }
                if (mpkt->readCallback) { //just for debug, test if mpkt is correctly assigned with readback function
                    mpkt->readCallback(mpkt->getPtr<uint8_t>(), mpkt->getSize());
                } else {
                    panic("readCallback is not set");
                }
            }
            delete mpkt;
        }
    }
}


bool MatrixMMU::writeMatrixMem(Addr addr, uint8_t* data, uint32_t size, ThreadContext *tc, uint8_t port_id, std::function<void(void)> writeCallback)
{
    if (matrix_PendingReqQs[port_id].size() >= pending_depth) {
        // 队列已满，无法发送请求
        return false;
    }
    uint64_t req_id = (uniqueReqId++);
    MemCmd cmd = MemCmd::WriteReq;
    const Addr pc = tc->pcState().instAddr();
    RequestPtr req = std::make_shared<Request>(addr, size, 0, MatrixCacheRequestorIds[port_id], pc, tc->contextId());
    MatrixPacketPtr mptr = new MatrixPacket(req, cmd, req_id, w_channel, writeCallback);
    // MatrixPacketPtr mptr = std::make_shared<MatrixPacket>(req, cmd, req_id, w_channel, writeCallback);
    matrix_PendingReqQs[port_id].push_back(mptr);
    if(!matrix_mem_ports[port_id].sendTimingWriteReq(addr, data, size, tc, req_id, w_channel, port_id)){
        delete matrix_PendingReqQs[port_id].back();
        matrix_PendingReqQs[port_id].pop_back();
        return false;
    }
    return true;
}

bool MatrixMMU::readMatrixMem(Addr addr, uint32_t size, ThreadContext *tc, uint8_t port_id, uint8_t dst_reg, std::function<void(uint8_t* data, uint8_t size)> readCallback)
{
    if (matrix_PendingReqQs[port_id].size() >= pending_depth) {
        // 队列已满，无法发送请求
        return false;
    }
    uint64_t req_id = (uniqueReqId++);
    MemCmd cmd = MemCmd::ReadReq;
    const Addr pc = tc->pcState().instAddr();
    RequestPtr req = std::make_shared<Request>(addr, size, 0, MatrixCacheRequestorIds[port_id], pc, tc->contextId());
    MatrixPacketPtr mptr = new MatrixPacket(req, cmd, req_id, r_channel, dst_reg, readCallback);
    // MatrixPacketPtr mptr = std::make_shared<MatrixPacket>(req, cmd, req_id, r_channel, readCallback);
    matrix_PendingReqQs[port_id].push_back(mptr);
    if(!matrix_mem_ports[port_id].sendTimingReadReq(addr,size, tc, req_id, r_channel, port_id)){
        delete matrix_PendingReqQs[port_id].back();
        matrix_PendingReqQs[port_id].pop_back();
        return false;
    }
    return true;
}

bool MatrixMMU::isFull()
{
    // assert(matrix_PendingReqQs[0].size()!=pending_depth);
    return(matrix_PendingReqQs[0].size()==pending_depth);
}

Port& MatrixMMU::getPort(const std::string &if_name, PortID idx)
{
    if(if_name == "matrix_mem_ports"){
        return getMatrixMemPort(idx);
    } else {
        return getPort(if_name, idx);
    }
}

void MatrixMMU::set_cpu_ptr(gem5::o3::CPU* _o3cpu)
{
    o3cpu = _o3cpu;
}

void MatrixMMU::set_matrixEngine_ptr(MatrixEngine* _matrix_engine)
{
    matrix_engine = _matrix_engine;
}
} //namespace gem5