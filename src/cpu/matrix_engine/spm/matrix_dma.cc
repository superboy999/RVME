#include "cpu/matrix_engine/spm/matrix_dma.hh"
#include "cpu/matrix_engine/spm/matrix_spm.hh"
#include "debug/MatrixDMA.hh"
#include "debug/MatrixDMAPort.hh"

namespace gem5
{
MatrixDmaDevice::MatrixDmaDevice(const MatrixDmaDeviceParams &params) :
    TickedObject(params, Event::Serialize_Pri),
    dma_buffer_width(params.dma_buffer_width),
    dma_buffer_depth(params.dma_buffer_depth),
    dma_transpose_buffer_num(params.dma_transpose_buffer_num),
    dma_transpose_buffer_width(params.dma_transpose_buffer_width),
    dma_transpose_buffer_depth(params.dma_transpose_buffer_depth),
    cache_line_size(params.cache_line_size),
    dmaPort(name() + ".dmaPort", *this, cache_line_size),
    dma_buffer(1, dma_buffer_width, dma_buffer_depth, cache_line_size, "DmaBuffer"),
    dma_transpose_buffer(dma_transpose_buffer_num, dma_transpose_buffer_width, dma_transpose_buffer_depth, cache_line_size, "DmaTransposeBuffer")
{
    // dma_buffer.resize(dma_buffer_depth, std::vector<uint8_t>(dma_buffer_width, 0));
    // dma_buffer_ready.resize(dma_buffer_depth, std::vector<bool>(dma_buffer_width, false));
    // dma_transpose_buffer.resize(dma_transpose_buffer_num, std::vector<std::vector<uint8_t>>(dma_buffer_width, std::vector<uint8_t>(dma_buffer_depth, 0)));
    // dma_transpose_buffer_ready.resize(dma_transpose_buffer_num, std::vector<std::vector<bool>>(dma_buffer_width, std::vector<bool>(dma_buffer_depth, false)));
    // MatrixSPMRequestorId = params.system->getRequestorId(this, name() + ".matrix_spm");
    MatrixSPMRequestorId = 3;
    dmaReqId = 0;
}

MatrixDmaDevice::~MatrixDmaDevice()
{}

void
MatrixDmaDevice::set_spm_ptr(MatrixSPM *spm)
{
    matrix_spm = spm;
    dmaAccessSpmQueue.resize(matrix_spm->sramBanks * 3);
}

Port&
MatrixDmaDevice::getPort(const std::string &if_name, PortID idx)
{
    if (if_name == "dmaPort") {
        return dmaPort;
    } else {
        return ClockedObject::getPort(if_name, idx);
    }
}

// void
// MatrixDmaDevice::addReq(MemCmd cmd, Addr spmAddr, Addr memAddr, uint32_t totBytes)
// {
//     uint64_t reqId = dmaReqId++;
//     bool needtranspose = (matrix_spm->decodeAddr(spmAddr).setid == 0) && cmd == MemCmd::ReadReq; // Use transpose buffer only for load data on setid 0
//     if(needtranspose) {
//         assert(totBytes % (dma_transpose_buffer_width * dma_transpose_buffer_depth) == 0);
//     }
//     dmaReqQueue.push_back(new DmaReqState(cmd, spmAddr, memAddr, totBytes, reqId, needtranspose));
//     DPRINTF(MatrixDMA, "Added DMA request: cmd %s, spmAddr %#llx, memAddr %#llx, totBytes %u, reqId %llu\n",
//             cmd.toString(), spmAddr, memAddr, totBytes, reqId);
//     // Process the DMA request immediately
//     startTicking();
// }

uint64_t
MatrixDmaDevice::addReq(MemCmd cmd, Addr spmAddr, Addr memAddr, uint32_t totBytes, std::shared_ptr<DmaReqState> shared_ptr)
{
    uint64_t reqId = dmaReqId++;
    bool needtranspose = (matrix_spm->decodeAddr(spmAddr).setid == 0) && cmd == MemCmd::ReadReq; // Use transpose buffer only for load data on setid 0
    // if(needtranspose) {
    //     assert(totBytes % (dma_transpose_buffer_width * dma_transpose_buffer_depth) == 0);
    // }
    shared_ptr->isDMA = true;
    shared_ptr->isFence = false;
    shared_ptr->totBytes = totBytes;
    shared_ptr->needtranspose = needtranspose;
    shared_ptr->reqId = reqId;
    shared_ptr->fence = matrix_spm->global_fence;
    dmaReqQueue.push_back(shared_ptr);
    DPRINTF(MatrixDMA, "Added request to DMA queue: reqId %llu, cmd %s, spmAddr %#llx, memAddr %#llx, totBytes %u, fence %llu\n",
            shared_ptr->reqId, shared_ptr->cmd.toString(), shared_ptr->spmAddr, shared_ptr->memAddr, shared_ptr->totBytes, shared_ptr->fence);
    startTicking();
    return shared_ptr->reqId;
}

uint64_t
MatrixDmaDevice::addReq(std::shared_ptr<DmaReqState> shared_ptr, uint8_t setid, bool isread)
{
    uint64_t reqId = dmaReqId++;
    shared_ptr->isDMA = false;
    shared_ptr->isFence = false;
    shared_ptr->reqId = reqId;
    shared_ptr->fence = matrix_spm->global_fence;
    dmaReqQueue.push_back(shared_ptr);
    DPRINTF(MatrixDMA, "Added request to DMA queue: reqId %llu, fence %llu, isDMA %d, isFence %d\n", shared_ptr->reqId, shared_ptr->fence, shared_ptr->isDMA, shared_ptr->isFence);
    matrix_spm->addToReqQueue(shared_ptr->reqId, setid, isread);
    startTicking();
    return shared_ptr->reqId;
}

uint64_t
MatrixDmaDevice::addSync(std::shared_ptr<DmaReqState> shared_ptr)
{
    uint64_t reqId = dmaReqId++;
    shared_ptr->reqId = reqId;
    shared_ptr->fence = ++matrix_spm->global_fence;
    dmaReqQueue.push_back(shared_ptr);
    DPRINTF(MatrixDMA, "Added request to DMA queue: reqId %llu, fence %llu, isDMA %d, isFence %d\n", shared_ptr->reqId, shared_ptr->fence, shared_ptr->isDMA, shared_ptr->isFence);
    // Process the DMA request immediately
    startTicking();
    return shared_ptr->reqId;
}

void
MatrixDmaDevice::startTicking()
{
    start();
    DPRINTF(MatrixDMA, "MatrixDmaDevice started ticking.\n");
}

void
MatrixDmaDevice::stopTicking()
{
    stop();
    DPRINTF(MatrixDMA, "MatrixDmaDevice stopped ticking.\n");
}

void
MatrixDmaDevice::regStats() 
{
    TickedObject::regStats();
    dma_read_bytes
        .name(name() + ".dmareadbytes")
        .desc("Count how many bytes dma read");
    dma_write_bytes
        .name(name() + ".dmawritebytes")
        .desc("Count how many bytes dma write");
}

void
MatrixDmaDevice::evaluate()
{
    assert(matrix_spm != nullptr);
    accessSPM();
    Issue();
    scheduleAccessSPM();
    if(dmaReqQueue.empty()) {
        stopTicking();
    }
}

void
MatrixDmaDevice::accessSPM()
{
    // Process the SPM access requests in the queue
    bool empty = true;
    for (auto queue : dmaAccessSpmQueue) {
        if (!queue.empty()) {
            empty = false;
            break;
        }
    }
    if (empty) {
        DPRINTF(MatrixDMA, "accessSPM: No DMA access SPM requests to process\n");
        return;
    }
    for (auto &queue : dmaAccessSpmQueue) {
        if (!queue.empty()) {
            auto &entry = queue.front();
            uint8_t *data = new uint8_t[entry.size];
            if (entry.isread) {               
                // Read data from SPM to DMA buffer
                DPRINTF(MatrixDMA, "accessSPM: Reading data from SPM to DMA buffer: spm_addr %#llx, buffer_addr %#llx, size %u\n",
                        entry.spm_addr, entry.buffer_addr, entry.size);
                matrix_spm->readSPM(entry.spm_addr, data, entry.size);
                dma_buffer.write(entry.buffer_addr, data, entry.size);
                dma_buffer.setReady(entry.buffer_addr, entry.size);
                assert(entry.dmaReqState != nullptr);
                entry.dmaReqState->readyBytes += entry.size;
            } else {
                // Write data from DMA buffer to SPM
                if(!entry.istranspose) {
                    DPRINTF(MatrixDMA, "accessSPM: Writing data from DMA buffer to SPM: spm_addr %#llx, buffer_addr %#llx, size %u\n",
                            entry.spm_addr, entry.buffer_addr, entry.size);
                    dma_buffer.read(entry.buffer_addr, data, entry.size);
                    matrix_spm->writeSPM(entry.spm_addr, data, entry.size);
                    entry.dmaReqState->finishedBytes += entry.size;
                    // dma_buffer.resetReady(entry.buffer_addr, entry.size);
                    // dma_buffer.resetOccupied(entry.buffer_addr, entry.size);
                } else {
                    DPRINTF(MatrixDMA, "accessSPM: Writing transposed data from DMA Transpose buffer to SPM: spm_addr %#llx, buffer_addr %#llx, size %u\n",
                            entry.spm_addr, entry.buffer_addr, entry.size);
                    dma_transpose_buffer.readCol(entry.transposebuffer_num, entry.transposebuffer_col, data, entry.size);
                    matrix_spm->writeSPM(entry.spm_addr, data, entry.size);
                    entry.dmaReqState->finishedBytes += entry.size;
                    // dma_transpose_buffer.resetReady(entry.buffer_addr, entry.size);
                    // dma_transpose_buffer.resetOccupied(entry.buffer_addr, entry.size);
                }
                // if(entry.dmaReqState->spmAccessBytes == entry.dmaReqState->totBytes) {
                //     // If all bytes are processed, remove the request from the queue
                //     DPRINTF(MatrixDMA, "accessSPM: All bytes processed for DMA request %llu, removing from queue.\n", entry.dmaReqState->reqId);
                //     auto new_end = std::remove_if(dmaReqQueue.begin(), dmaReqQueue.end(),
                //         [&](const std::shared_ptr<DmaReqState>& req_ptr) {
                //             return req_ptr.get() == entry.dmaReqState;
                //         });
                //     dmaReqQueue.erase(new_end, dmaReqQueue.end());
                // }
            }
            delete[] data;
            queue.pop_front();
        }
    }
    // dma_buffer.printBuffer();
    // dma_transpose_buffer.printBuffer();
    // matrix_spm->printSpmData(0, 64); 
}

void
MatrixDmaDevice::Issue()
{
    if(dmaReqQueue.empty()) {
        DPRINTF(MatrixDMA, "Issue: No DMA requests to issue.\n");
        return;
    }
    assert(!dmaReqQueue.empty());
    for (auto it = dmaReqQueue.begin(); it != dmaReqQueue.end();) {
        const auto &reqState = *it; // shared_ptr<DmaReqState>
        if(reqState->isFence) {
            if(it == dmaReqQueue.begin()) {
                DPRINTF(MatrixDMA, "Issue: DMA request %llu is a fence, and it is the head of the queue. Pop it.\n", reqState->reqId);
                matrix_spm->fence_threshold++;
                it = dmaReqQueue.erase(it);
                continue;
            } else {
                DPRINTF(MatrixDMA, "Issue: DMA request %llu is a fence, but it is not the head of the queue. Stop issuing.\n", reqState->reqId);
                return;
            }
        }
        if(!reqState->isDMA) {
            DPRINTF(MatrixDMA, "Issue: DMA request %llu is a ME ld/st SPM instruction.\n", reqState->reqId);
            if(reqState->isExecuted) {
                DPRINTF(MatrixDMA, "Issue: DMA request %llu has been executed, pop it from the queue.\n", reqState->reqId);
                // Remove the request from the queue
                it = dmaReqQueue.erase(it);
                continue;
            } else {
                DPRINTF(MatrixDMA, "Issue: DMA request %llu has not been executed, waiting.\n", reqState->reqId);
                it++;
                continue;
            }
        }
        if(reqState->cmd == MemCmd::ReadReq) { // load data from memory to SPM
            DmaBuffer &buffer = reqState->needtranspose ? dma_transpose_buffer : dma_buffer;
            if(reqState->finishedBytes == reqState->totBytes) {
                DPRINTF(MatrixDMA, "Issue: All bytes finished for DMA request %llu, removing from queue.\n", reqState->reqId);
                buffer.resetReady(reqState->bufferAddr, reqState->totBytes);
                buffer.resetOccupied(reqState->bufferAddr, reqState->totBytes);
                it = dmaReqQueue.erase(it);
                matrix_spm->meminst_num--;
                continue;
            }
            if(reqState->issuedBytes >= reqState->totBytes) {
                assert(reqState->issuedBytes == reqState->totBytes);
                DPRINTF(MatrixDMA, "Issue: DMA request %llu already issued all bytes.\n", reqState->reqId);
                it++;
                continue;
            }
            PacketPtr pkt;
            if(dmaPort.inRetry && !dmaPort.retryiscalled) {
                DPRINTF(MatrixDMA, "Issue: Waiting memory hierarchy call retry request %llu\n", reqState->reqId);
                return;
            } else if (dmaPort.inRetry && dmaPort.retryiscalled) {
                DPRINTF(MatrixDMA, "Issue: Issuing retry request %llu\n", reqState->reqId);
                pkt = dmaPort.inRetry;           
            } else {
                Addr memAddr = reqState->memAddr + reqState->issuedBytes;
                // fixme: size 应该由指令传入，而不是每次都是cachelinesize
                uint32_t size = std::min(reqState->totBytes - reqState->issuedBytes, dmaPort.cache_line_size);
                if(!buffer.findFree(size)) {
                    DPRINTF(MatrixDMA, "Issue: No free space in buffer for request %llu, waiting for size %u space.\n", reqState->reqId, size);
                    return;
                }
                Addr bufferAddr;
                if(reqState->issuedBytes == 0) {
                    bufferAddr = buffer.getFreeAddr();
                    reqState->bufferAddr = bufferAddr;
                    DPRINTF(MatrixDMA, "Issue: DMA request %llu bufferAddr set to %#llx\n", reqState->reqId, bufferAddr);
                } else {
                    bufferAddr = (reqState->bufferAddr + reqState->issuedBytes) % buffer.totalSize;
                    assert(buffer.isFree(bufferAddr, size));
                }
                

                // //To ensure that a single access does not exceed the range of one page table.
                // Process *p = reqState->tc.getProcessPtr();
                // Addr page1 = p->pTable->pageAlign(memAddr);
                // Addr page2 = p->pTable->pageAlign(memAddr+size-1);
                // assert(page1 == page2);
                RequestPtr req;
                // if (memAddr < 0xfa000) {
                const Addr pc = reqState->tc.pcState().instAddr();
                req = std::make_shared<Request>(memAddr, size, 0, 0, pc, reqState->tc.contextId());
                if (auto *p = reqState->tc.getProcessPtr()) {
                    Addr paddr;
                    if (!p->pTable->translate(memAddr, paddr)) {
                        panic("DMA: SE page table has no mapping for vaddr=%#lx\n", memAddr);
                    }
                    req->setPaddr(paddr);
                }
                // DMA_Translation* translation = new DMA_Translation();
                // DPRINTF(MatrixDMA, "Issue: before translation addr = %lx\n", memAddr);
                // matrix_spm->matrix_engine->o3cpu->mmu->dtb->translateTiming(req, &reqState->tc, translation, BaseMMU::Read);
                // DPRINTF(MatrixDMA, "Issue: after translation addr = %lx\n", req->getPaddr());
                // if(translation->fault != NoFault) {
                //     delete translation;
                //         panic("DMA request %llu translate address %#llx failed with %s\n", reqState->reqId, memAddr, translation->fault->name());
                // } else {
                //     delete translation;
                // }
                // } else {
                // req = std::make_shared<Request>(memAddr - 0x10000, size, 0, 0);
                // }
                // pkt = new DmaPacket(req, reqState->cmd, reqState->needtranspose, bufferAddr, reqState->reqId);
                pkt = new Packet(req, MemCmd::ReadReq);
                uint8_t *ndata = new uint8_t[size];
                memset(ndata, 'Z', size);
                pkt->dataDynamic(ndata);
                pkt->senderState = new DmaSenderState(reqState->reqId, reqState->needtranspose, bufferAddr, size);
                DPRINTF(MatrixDMA, "Issue: Issuing DMA %s pkt belong to request %llu: memAddr %#llx, bufferAddr %#llx, size %u\n",
                        pkt->cmd.toString(), reqState->reqId, req->getPaddr(), bufferAddr, pkt->getSize());
                buffer.setOccupied(bufferAddr, size);
                buffer.resetReady(bufferAddr, size);
                buffer.printReady();
                buffer.printOccupied();
            }
            dmaPort.inRetry = nullptr;
            assert(pkt != nullptr);
            if(dmaPort.sendTimingReq(pkt)) {
                reqState->issuedBytes += pkt->getSize();
                DPRINTF(MatrixDMA, "Issue: DMA request %llu issue pkt successfully. Now issuedBytes = %llu\n", reqState->reqId, reqState->issuedBytes);
            } else {
                DPRINTF(MatrixDMA, "Issue: DMA request %llu issue pkt failed, retrying later.\n", reqState->reqId);
                dmaPort.inRetry = pkt;
                dmaPort.retryiscalled = false;
            }
            return;
        } else if (reqState->cmd == MemCmd::WriteReq) { // store data from SPM to memory
            if (reqState->issuedBytes < reqState->readyBytes) {
                PacketPtr pkt;
                if(dmaPort.inRetry && !dmaPort.retryiscalled) {
                    DPRINTF(MatrixDMA, "Issue: Waiting memory hierarchy call retry request %llu\n", reqState->reqId);
                    return;
                } else if (dmaPort.inRetry && dmaPort.retryiscalled) {
                    DPRINTF(MatrixDMA, "Issue: Issuing retry request %llu\n", reqState->reqId);
                    pkt = dmaPort.inRetry;
                } else {
                    uint32_t readyBytes = reqState->readyBytes - reqState->issuedBytes;
                    if(readyBytes < dmaPort.cache_line_size) {
                        DPRINTF(MatrixDMA, "Issue: Not enough ready bytes (%u < %u) for request %llu, waiting for more data.\n", readyBytes, dmaPort.cache_line_size, reqState->reqId);
                        return;
                    }
                    Addr memAddr = reqState->memAddr + reqState->issuedBytes;
                    uint32_t size = dmaPort.cache_line_size;
                    Addr bufferAddr = (reqState->bufferAddr + reqState->issuedBytes) % dma_buffer.totalSize;
                    
                    RequestPtr req;
                    // if(memAddr < 0xfa000) {
                    const Addr pc = reqState->tc.pcState().instAddr();
                    req = std::make_shared<Request>(memAddr, size, 0, 0, pc, reqState->tc.contextId());
                    // DMA_Translation* translation = new DMA_Translation();
                    // DPRINTF(MatrixDMA, "Issue: before translation addr = %lx\n", memAddr);
                    // matrix_spm->matrix_engine->o3cpu->mmu->dtb->translateTiming(req, &reqState->tc, translation, BaseMMU::Write);
                    // DPRINTF(MatrixDMA, "Issue: after translation addr = %lx\n", req->getPaddr());
                    if (auto *p = reqState->tc.getProcessPtr()) {
                        Addr paddr;
                        if (!p->pTable->translate(memAddr, paddr)) {
                            panic("DMA: SE page table has no mapping for vaddr=%#lx\n", memAddr);
                        }
                        req->setPaddr(paddr);
                    }
                    // if(translation->fault != NoFault) {
                    //     delete translation;
                    //         panic("DMA request %llu translate address %#llx failed with %s\n", reqState->reqId, memAddr, translation->fault->name());
                    // } else {
                    //     delete translation;
                    // }     
                    // } else {
                        // req = std::make_shared<Request>(memAddr - 0x10000, size, 0, 0);
                    // }
                    // pkt = new DmaPacket(req, reqState->cmd, reqState->needtranspose, bufferAddr, reqState->reqId);
                    pkt = new Packet(req, MemCmd::WriteReq);
                    uint8_t *ndata = new uint8_t[size];
                    dma_buffer.read(bufferAddr, ndata, size);
                    pkt->dataDynamic(ndata);
                    pkt->senderState = new DmaSenderState(reqState->reqId, reqState->needtranspose, bufferAddr, size);               
                    DPRINTF(MatrixDMA, "Issue: Issuing DMA %s pkt belong to request %llu: memAddr %#llx, bufferAddr %#llx, size %u\n",
                            pkt->cmd.toString(), reqState->reqId, req->getPaddr(), bufferAddr, pkt->getSize()); 
                    // printf("Issue: Issuing DMA %s pkt belong to request %llu: memAddr %#llx, bufferAddr %#llx, size %u\n",
                    //         pkt->cmd.toString(), reqState->reqId, req->getPaddr(), bufferAddr, pkt->getSize());                 
                }
                dmaPort.inRetry = nullptr;
                assert(pkt != nullptr);
                if(dmaPort.sendTimingReq(pkt)) {   
                    reqState->issuedBytes += pkt->getSize();
                    DPRINTF(MatrixDMA, "Issue: DMA request %llu issue pkt successfully. Now issuedBytes = %llu\n", reqState->reqId, reqState->issuedBytes);
                    // dma_buffer.resetOccupied(reqState->bufferAddr + reqState->issuedBytes - pkt->getSize(), pkt->getSize());
                } else {
                    DPRINTF(MatrixDMA, "Issue: DMA request %llu issue pkt failed, retrying later.\n", reqState->reqId);
                    dmaPort.inRetry = pkt;
                    dmaPort.retryiscalled = false;
                }
                return;
            }
        } else {
            panic("Issue: Unknown DMA command %s", reqState->cmd.toString());
        }
        ++it;
    }
}

// scheduleAccessSPM用于将新ready的数据试图发送写SPM或者发送新的读取SPM
// 每个bank每个周期都可以处理一个访问请求
// 在下一个周期的AccessSPM()即可看到这个访问请求从而实现延时处理
void
MatrixDmaDevice::scheduleAccessSPM()
{
    if(dmaReqQueue.empty()) {
        DPRINTF(MatrixDMA, "scheduleAccessSPM: No DMA requests to schedule.\n");
        return;
    }
    assert(!dmaReqQueue.empty());
    for (auto it = dmaReqQueue.begin(); it != dmaReqQueue.end();) {
        const auto &reqState = *it; // shared_ptr<DmaReqState>
        if(reqState->isFence) {
            DPRINTF(MatrixDMA, "scheduleAccessSPM: DMA request %llu is a fence. Stop scheduleAccessSPM.\n", reqState->reqId);
            return;
        }
        if(!reqState->isDMA) {
            DPRINTF(MatrixDMA, "scheduleAccessSPM: DMA request %llu is a ME ld/st SPM instruction, skip it.\n", reqState->reqId);
            it++;
            continue;
        }
        if(reqState->cmd == MemCmd::ReadReq){ // load data from memory to SPM
            if(reqState->spmAccessBytes < reqState->readyBytes) { // 需要从DMA buffer写回SPM
                if (!reqState->needtranspose) {
                    uint32_t size = reqState->readyBytes - reqState->spmAccessBytes;
                    Addr spmAddr = reqState->spmAddr + reqState->spmAccessBytes;
                    Addr bufferAddr = reqState->bufferAddr + reqState->spmAccessBytes;
                    assert(size > 0);
                    assert(matrix_spm->decodeAddr(spmAddr).setid == matrix_spm->decodeAddr(spmAddr + size - 1).setid);
                    unsigned entryWidth = matrix_spm->entryWidth;
                    unsigned offset = spmAddr & (entryWidth - 1);
                    unsigned int b_pkt_count = divCeil(offset + size, entryWidth);
                    Addr spm_addr = spmAddr;
                    uint32_t buffer_addr = bufferAddr;
                    for (int cnt = 0; cnt < b_pkt_count; ++cnt) {
                        unsigned entry_size = std::min((spm_addr | (entryWidth - 1)) + 1,
                                            spmAddr + size) - spm_addr;
                        MatrixSPM::BankEntry bank_entry = matrix_spm->decodeAddr(spm_addr);
                        // dmaAccessSpmEntry entry(false, false, spm_addr, buffer_addr, entry_size, reqState);
                        DPRINTF(MatrixDMA, "scheduleAccessSPM: create a dmaAccessSpmEntry with spm_addr %#llx, buffer_addr %#llx, size %u"
                            ", isRead %d, needtranspose %d\n", spm_addr, buffer_addr, entry_size, false, false);
                        // dmaAccessSpmQueue[bank_entry.bankid].push_back(entry);
                        dmaAccessSpmQueue[bank_entry.bankid].emplace_back(false, false, spm_addr, buffer_addr, entry_size, reqState);
                        // printf("[wrp] dmaAccessSpmQueue[%u].size = %lu\n", bank_entry.bankid, dmaAccessSpmQueue[bank_entry.bankid].size());
                        spm_addr += entry_size;
                        buffer_addr += entry_size;
                        buffer_addr = buffer_addr % dma_buffer.totalSize;
                    }
                    reqState->spmAccessBytes += size;
                } else {
                    uint32_t readySize = reqState->readyBytes - reqState->spmAccessBytes;
                    uint32_t transposeBufferSize = dma_transpose_buffer_width * dma_transpose_buffer_depth;
                    if(readySize < transposeBufferSize) {
                        DPRINTF(MatrixDMA, "scheduleAccessSPM: Not enough data in DMA transpose buffer (%u < %u) for request %llu, waiting for more data.\n", readySize, transposeBufferSize, reqState->reqId);
                        return;
                    }
                    uint32_t buffer_num = readySize / transposeBufferSize;
                    for (uint32_t i = 0; i < buffer_num; ++i) {
                        Addr bufferAddr = (reqState->bufferAddr + reqState->spmAccessBytes + i * transposeBufferSize) % dma_transpose_buffer.totalSize;
                        assert(bufferAddr % transposeBufferSize == 0);
                        assert(dma_transpose_buffer.isReady(bufferAddr, transposeBufferSize));
                        DmaBufferEntry bufferEntry = dma_transpose_buffer.decodeAddr(bufferAddr);
                        uint32_t colIdx = dma_transpose_buffer.transposeState[bufferEntry.buffer_num];
                        uint32_t size = dma_transpose_buffer_depth;
                        Addr spmAddr = reqState->spmAddr + reqState->spmAccessBytes + colIdx * dma_transpose_buffer_depth + i * transposeBufferSize;
                        unsigned entryWidth = matrix_spm->entryWidth;
                        unsigned offset = spmAddr & (entryWidth - 1);
                        unsigned int b_pkt_count = divCeil(offset + size, entryWidth);
                        Addr buffer_addr = bufferAddr + colIdx * dma_transpose_buffer_depth;
                        Addr spm_addr = spmAddr;
                        // fixme: 目前只支持entryWidth = dma_transpose_buffer_depth的情况
                        assert(size == entryWidth);
                        assert(b_pkt_count == 1); 
                        for (int cnt = 0; cnt < b_pkt_count; ++cnt) {
                            unsigned entry_size = std::min((spm_addr | (entryWidth - 1)) + 1,
                                                spmAddr + size) - spm_addr;
                            MatrixSPM::BankEntry bank_entry = matrix_spm->decodeAddr(spm_addr);
                            // dmaAccessSpmEntry entry(false, false, spm_addr, buffer_addr, entry_size, reqState);
                            // entry.transposebuffer_num = bufferEntry.buffer_num;
                            // entry.transposebuffer_col = colIdx;
                            DPRINTF(MatrixDMA, "scheduleAccessSPM: create a dmaAccessSpmEntry with spm_addr %#llx, buffer_addr %#llx, size %u"
                                ", isRead %d, needtranspose %d\n", spm_addr, buffer_addr, entry_size, false, true);
                            // dmaAccessSpmQueue[bank_entry.bankid].push_back(entry);
                            dmaAccessSpmQueue[bank_entry.bankid].emplace_back(false, true, spm_addr, buffer_addr, entry_size, bufferEntry.buffer_num, colIdx, reqState);
                            // printf("[wrp] dmaAccessSpmQueue[%u].size = %lu\n", bank_entry.bankid, dmaAccessSpmQueue[bank_entry.bankid].size());
                            spm_addr += entry_size;
                        }
                        dma_transpose_buffer.transposeState[bufferEntry.buffer_num]++;
                        if(dma_transpose_buffer.transposeState[bufferEntry.buffer_num] >= dma_transpose_buffer_width) {
                            reqState->spmAccessBytes += transposeBufferSize;
                            dma_transpose_buffer.transposeState[bufferEntry.buffer_num] = 0;
                        }
                    }
                }
            }
        } else if (reqState->cmd == MemCmd::WriteReq) { // store data from SPM to memory
            if(reqState->spmAccessBytes >= reqState->totBytes) {
                assert(reqState->spmAccessBytes == reqState->totBytes);
                DPRINTF(MatrixDMA, "scheduleAccessSPM: DMA request %llu already scheduled all bytes access from SPM.\n", reqState->reqId);
                it++;
                continue;
            }
            uint32_t size = std::min(reqState->totBytes - reqState->spmAccessBytes, dmaPort.cache_line_size);
            Addr spmAddr = reqState->spmAddr + reqState->spmAccessBytes;
            assert(size > 0);
            assert(matrix_spm->decodeAddr(spmAddr).setid == matrix_spm->decodeAddr(spmAddr + size - 1).setid);         
            if(!dma_buffer.findFree(size)) {
                DPRINTF(MatrixDMA, "scheduleAccessSPM: No free space in buffer for request %llu, waiting for size %u space.\n", reqState->reqId, size);
                return;
            }
            Addr bufferAddr;
            if(reqState->spmAccessBytes == 0) {        
                bufferAddr = dma_buffer.getFreeAddr();
                assert(dma_buffer.isFree(bufferAddr, size));
                reqState->bufferAddr = bufferAddr;
                DPRINTF(MatrixDMA, "scheduleAccessSPM: DMA request %llu bufferAddr set to %#llx\n", reqState->reqId, bufferAddr);
            } else {
                bufferAddr = reqState->bufferAddr + reqState->spmAccessBytes;
                assert(dma_buffer.isFree(bufferAddr, size));
            }
            unsigned entryWidth = matrix_spm->entryWidth;
            unsigned offset = spmAddr & (entryWidth - 1);
            unsigned int b_pkt_count = divCeil(offset + size, entryWidth);
            Addr spm_addr = spmAddr;
            Addr buffer_addr = bufferAddr;
            for (int cnt = 0; cnt < b_pkt_count; ++cnt) {
                unsigned entry_size = std::min((spm_addr | (entryWidth - 1)) + 1,
                                    spmAddr + size) - spm_addr;
                MatrixSPM::BankEntry bank_entry = matrix_spm->decodeAddr(spm_addr);
                // dmaAccessSpmEntry entry(true, false, spm_addr, buffer_addr, entry_size, reqState);
                DPRINTF(MatrixDMA, "scheduleAccessSPM: create a dmaAccessSpmEntry with spm_addr %#llx, buffer_addr %#llx, size %u"
                    ", isRead %d, needtranspose %d\n", spm_addr, buffer_addr, entry_size, true, false);
                // dmaAccessSpmQueue[bank_entry.bankid].push_back(entry);
                dmaAccessSpmQueue[bank_entry.bankid].emplace_back(true, false, spm_addr, buffer_addr, entry_size, reqState);
                // printf("[wrp] dmaAccessSpmQueue[%u].size = %lu\n", bank_entry.bankid, dmaAccessSpmQueue[bank_entry.bankid].size());
                spm_addr += entry_size;
                buffer_addr += entry_size;
                buffer_addr = buffer_addr % dma_buffer.totalSize;
            }
            reqState->spmAccessBytes += size;           
            dma_buffer.setOccupied(bufferAddr, size);
            dma_buffer.resetReady(bufferAddr, size);
            // dma_buffer.printReady();
            // dma_buffer.printOccupied();
        } else {
            panic("Unknown DMA command %s", reqState->cmd.toString());
        }
        ++it;
    }
}

MatrixDmaDevice::DmaPort::DmaPort(const std::string& _name, MatrixDmaDevice& _device, uint32_t _cache_line_size)
    : RequestPort(_name), device(_device), cache_line_size(_cache_line_size)
{ }

MatrixDmaDevice::DmaPort::~DmaPort()
{
    if (inRetry) {
        delete inRetry;
        inRetry = nullptr;
    }
}

bool
MatrixDmaDevice::DmaPort::recvTimingResp(PacketPtr pkt)
{
    assert(pkt->req->isUncacheable() ||
           !(pkt->cacheResponding() && !pkt->hasSharers()));
    // DmaPacketPtr dma_pkt = dynamic_cast<DmaPacketPtr>(pkt);
    // assert(dma_pkt != nullptr);
    // if(dma_pkt->isRead()) {
    //     DmaBuffer &buffer = dma_pkt->needtranspose ? device.dma_transpose_buffer : device.dma_buffer;
    //     for (uint32_t i = 0; i < dma_pkt->getSize(); ++i) {
    //         // printf("[wrp] DMA Read Response: Received data[%u] = %02x for DMA pkt belong to request %llu\n",
    //                i, dma_pkt->getPtr<uint8_t>()[i], dma_pkt->reqId);
    //     }
    //     buffer.write(dma_pkt->bufferAddr, dma_pkt->getPtr<uint8_t>(), dma_pkt->getSize());
    //     buffer.setReady(dma_pkt->bufferAddr, dma_pkt->getSize());
    //     for (const auto& reqState : device.dmaReqQueue) {
    //         if (reqState->reqId == dma_pkt->reqId) {
    //             reqState->readyBytes += dma_pkt->getSize();
    //             // DPRINTF(MatrixDMA, "DMA Read completed for request %llu, readyBytes now %llu\n", reqState->reqId, reqState->readyBytes);
    //             break;
    //         }
    //     }
    // } else if(dma_pkt->isWrite()) {
    //     DPRINTF(MatrixDMA, "DMA Write %#llx completed\n", dma_pkt->getAddr());
    //     for (auto it = device.dmaReqQueue.begin(); it != device.dmaReqQueue.end();) {
    //         if ((*it)->reqId == dma_pkt->reqId) {
    //             if ((*it)->issuedBytes == (*it)->totBytes) {
    //                 // DPRINTF(MatrixDMA, "DMA Store instruction completed for request %llu, remove it\n", (*it)->reqId);
    //                 it = device.dmaReqQueue.erase(it);
    //                 device.matrix_spm->meminst_num--;
    //                 break;
    //             }
    //         }
    //         ++it;
    //     }
    // } else {
    //     panic("Unknown DMA packet command");
    // }

    auto *sender_state = safe_cast<DmaSenderState*>(pkt->senderState);
    assert(sender_state != nullptr);
    if(pkt->isRead()) {
        DmaBuffer &buffer = sender_state->needTranspose ? device.dma_transpose_buffer : device.dma_buffer;
        // buffer.printData();
        for (uint32_t i = 0; i < pkt->getSize(); ++i) {
            // printf("[wrp] DMA Read Response: Received data[%u] = %02x for DMA pkt belong to request %llu, buffer addr = %llu, size = %llu\n",
            //        i, pkt->getPtr<uint8_t>()[i], sender_state->reqId, sender_state->bufferAddr, pkt->getSize());
        }
        buffer.write(sender_state->bufferAddr, pkt->getPtr<uint8_t>(), pkt->getSize());
        // Print received packet data to stdout (minimal, non-intrusive):
        // only dump up to 64 bytes to avoid huge logs.
        // {
        //     const uint8_t *data_ptr = pkt->getPtr<uint8_t>();
        //     uint32_t total = pkt->getSize();
        //     uint32_t dump_len = total > 64 ? 64 : total;
        //          printf("DMA Read Response: %s pkt belong to request %llu: memAddr %#llx, bufferAddr %#llx, size %u\n",
        //              pkt->cmd.toString(),
        //              (unsigned long long)sender_state->reqId,
        //              (unsigned long long)pkt->req->getPaddr(),
        //              (unsigned long long)sender_state->bufferAddr,
        //              total);
        //     for (uint32_t i = 0; i < dump_len; ++i) {
        //         printf(" %02x", data_ptr[i]);
        //         if ((i + 1) % 16 == 0)
        //             printf("\n");
        //     }
        //     if (dump_len % 16 != 0)
        //         printf("\n");
        //     if (total > dump_len)
        //         printf("... (only first %u of %u bytes shown)\n", dump_len, total);
        // }
        device.dma_read_bytes += pkt->getSize();
        device.dma_read_req++;
        buffer.setReady(sender_state->bufferAddr, pkt->getSize());
        for (const auto& reqState : device.dmaReqQueue) {
            if (reqState->reqId == sender_state->reqId) {
                reqState->readyBytes += pkt->getSize();
                DPRINTF(MatrixDMA, "DMA Read completed for request %llu, readyBytes now %llu\n", reqState->reqId, reqState->readyBytes);
                break;
            }
        }
    } else if(pkt->isWrite()) {
        DPRINTF(MatrixDMA, "DMA Write %#llx completed\n", pkt->getAddr());
        device.dma_write_bytes += pkt->getSize();
        device.dma_write_req++;
        for (auto it = device.dmaReqQueue.begin(); it != device.dmaReqQueue.end();) {
            if ((*it)->reqId == sender_state->reqId) {
                (*it)->finishedBytes += pkt->getSize();
                if ((*it)->finishedBytes == (*it)->totBytes) {
                    DPRINTF(MatrixDMA, "DMA Store instruction completed for request %llu, remove it\n", (*it)->reqId);
                    device.dma_buffer.resetReady((*it)->bufferAddr, (*it)->totBytes);
                    device.dma_buffer.resetOccupied((*it)->bufferAddr, (*it)->totBytes);
                    it = device.dmaReqQueue.erase(it);
                    device.matrix_spm->meminst_num--;
                    break;
                }
            }
            ++it;
        }
    } else {
        panic("Unknown DMA packet command");
    }

    delete sender_state;
    pkt->senderState = nullptr;
    delete pkt;
    return true;
}

void
MatrixDmaDevice::DmaPort::recvReqRetry()
{
    retryiscalled = true;
}



MatrixDmaDevice::DmaBuffer::DmaBuffer(uint32_t _buffer_num, uint32_t _width, uint32_t _depth, uint32_t cache_line_size, const std::string & _name)
    : buffer_num(_buffer_num), width(_width), depth(_depth), cache_line_size(cache_line_size), name(_name)
{
    data.resize(buffer_num, std::vector<std::vector<uint8_t>>(depth, std::vector<uint8_t>(width, 0)));
    ready.resize(buffer_num, std::vector<std::vector<bool>>(depth, std::vector<bool>(width, false)));
    occupied.resize(buffer_num, std::vector<std::vector<bool>>(depth, std::vector<bool>(width, false)));
    totalSize = buffer_num * width * depth;
}

MatrixDmaDevice::DmaBufferEntry
MatrixDmaDevice::DmaBuffer::decodeAddr(Addr addr) const
{
    uint32_t buffer_num = addr / (width * depth);
    uint32_t row = (addr / width) % depth;
    uint32_t col = addr % width;
    return DmaBufferEntry(buffer_num, row, col);
}

void
MatrixDmaDevice::DmaBuffer::read(Addr addr, uint8_t* data_out, uint32_t size) const
{
    assert(size > 0);
    for(uint32_t i = 0; i < size; ++i) {
        if(addr == totalSize)
            addr = 0;
        DmaBufferEntry entry = decodeAddr(addr);
        assert(entry.buffer_num < buffer_num && entry.row < depth && entry.col < width);
        assert(ready[entry.buffer_num][entry.row][entry.col]);
        data_out[i] = data[entry.buffer_num][entry.row][entry.col];
        // printf("[wrp] Reading from %s at buffer %u, row %u, col %u, data %02x\n",
            //    name.c_str(), entry.buffer_num, entry.row, entry.col, data_out[i]);
        addr++;
    }
}

void
MatrixDmaDevice::DmaBuffer::write(Addr addr, const uint8_t* data_in, uint32_t size)
{
    assert(size > 0);
    for(uint32_t i = 0; i < size; ++i) {
        if(addr == totalSize)
            addr = 0;
        DmaBufferEntry entry = decodeAddr(addr);
        assert(entry.buffer_num < buffer_num && entry.row < depth && entry.col < width);
        data[entry.buffer_num][entry.row][entry.col] = data_in[i];
        addr++;
    }
}

bool
MatrixDmaDevice::DmaBuffer::findFree(uint32_t size) const
{
    uint32_t free_size = 0;
    for (uint32_t i = 0; i < totalSize; ++i) {
        DmaBufferEntry entry = decodeAddr(i);
        if (!occupied[entry.buffer_num][entry.row][entry.col]) {
            free_size++;
        }
    }
    if (free_size < size) {
        // printf("[wrp] %s has not enough space for size %u, "
            // "only %u not occupied\n", name.c_str(), size, free_size);
        return false;
    }
    return true;
}

Addr 
MatrixDmaDevice::DmaBuffer::getFreeAddr() const
{
    Addr min_not_occupied_addr = 0;
    bool find_min_not_occupied = false;
    Addr max_occupied_addr = 0;
    bool find_max_occupied = false;
    for (uint32_t i = 0; i < totalSize; ++i) {
        DmaBufferEntry entry = decodeAddr(i);
        if (!occupied[entry.buffer_num][entry.row][entry.col]) {
            if (!find_min_not_occupied) {
                min_not_occupied_addr = i;
                find_min_not_occupied = true;
            }
        } else {
            max_occupied_addr = i;
            find_max_occupied = true;
        }
    }
    Addr addr;
    if(!find_max_occupied) {
        addr = min_not_occupied_addr;
        assert(addr == 0);
    } else {
        if(max_occupied_addr == totalSize - 1 && find_min_not_occupied) {
            addr = min_not_occupied_addr;
        } else if (find_min_not_occupied) {
            addr = max_occupied_addr + 1;
        } else {
            addr = 0; // If no free space found, return 0
        }
    }
    return addr;
}

void 
MatrixDmaDevice::DmaBuffer::setOccupied(Addr addr, uint32_t size)
{
    for (uint32_t i = 0; i < size; ++i) {
        if(addr == totalSize)
            addr = 0;
        DmaBufferEntry entry = decodeAddr(addr);
        assert(entry.buffer_num < buffer_num && entry.row < depth && entry.col < width);
        assert(!occupied[entry.buffer_num][entry.row][entry.col]);
        occupied[entry.buffer_num][entry.row][entry.col] = true;
        // printf("[wrp] Setting occupied at %s %u, row %u, col %u\n",
            //    name.c_str(), entry.buffer_num, entry.row, entry.col);
        addr++;
    }
}

void 
MatrixDmaDevice::DmaBuffer::resetOccupied(Addr addr, uint32_t size)
{
    for (uint32_t i = 0; i < size; ++i) {
        if(addr == totalSize)
            addr = 0;
        DmaBufferEntry entry = decodeAddr(addr);
        assert(entry.buffer_num < buffer_num && entry.row < depth && entry.col < width);
        occupied[entry.buffer_num][entry.row][entry.col] = false;
        // printf("[wrp] Resetting occupied at %s %u, row %u, col %u as not occupied\n",
        //        name.c_str(), entry.buffer_num, entry.row, entry.col);
        addr++;
    }
}

void 
MatrixDmaDevice::DmaBuffer::setReady(Addr addr, uint32_t size)
{
    for (uint32_t i = 0; i < size; ++i) {
        if(addr == totalSize)
            addr = 0;
        DmaBufferEntry entry = decodeAddr(addr);
        assert(entry.buffer_num < buffer_num && entry.row < depth && entry.col < width);
        assert(!ready[entry.buffer_num][entry.row][entry.col]);
        ready[entry.buffer_num][entry.row][entry.col] = true;
        // printf("[wrp] Setting ready at %s %u, row %u, col %u\n",
        //        name.c_str(), entry.buffer_num, entry.row, entry.col);
        addr++;
    }
}

void 
MatrixDmaDevice::DmaBuffer::resetReady(Addr addr, uint32_t size)
{
    for (uint32_t i = 0; i < size; ++i) {
        if(addr == totalSize)
            addr = 0;
        DmaBufferEntry entry = decodeAddr(addr);
        assert(entry.buffer_num < buffer_num && entry.row < depth && entry.col < width);
        ready[entry.buffer_num][entry.row][entry.col] = false;
        // printf("[wrp] Resetting ready at %s %u, row %u, col %u as not ready\n",
        //        name.c_str(), entry.buffer_num, entry.row, entry.col);
        addr++;
    }
}

bool
MatrixDmaDevice::DmaBuffer::isFree(Addr addr, uint32_t size) const
{
    for (uint32_t i = 0; i < size; ++i) {
        if(addr == totalSize)
            addr = 0;
        DmaBufferEntry entry = decodeAddr(addr);
        assert(entry.buffer_num < buffer_num && entry.row < depth && entry.col < width);
        if (occupied[entry.buffer_num][entry.row][entry.col]) {
            return false;
        }
        addr++;
    }
    return true;
}

bool
MatrixDmaDevice::DmaBuffer::isReady(Addr addr, uint32_t size) const
{
    for (uint32_t i = 0; i < size; ++i) {
        if(addr == totalSize)
            addr = 0;
        DmaBufferEntry entry = decodeAddr(addr);
        assert(entry.buffer_num < buffer_num && entry.row < depth && entry.col < width);
        if (!ready[entry.buffer_num][entry.row][entry.col]) {
            return false;
        }
        addr++;
    }
    return true;
}

void
MatrixDmaDevice::DmaBuffer::printBuffer() const
{
    printData();
    printReady();
    printOccupied();
}

void
MatrixDmaDevice::DmaBuffer::printData() const
{
    // printf("========= %s Data Dump =========\n", name.c_str());
    for (uint32_t b = 0; b < buffer_num; ++b) {
        // printf("Buffer %u:\n", b);
        for (uint32_t r = 0; r < depth; ++r) {
            // printf("Row %2u: ", r);
            for (uint32_t c = 0; c < width; ++c) {
                // printf("%02x ", data[b][r][c]);
            }
            // printf("\n");
    }}
    // printf("=========================================\n");
}

void
MatrixDmaDevice::DmaBuffer::printReady() const
{
    // printf("========= %s Ready Dump =========\n", name.c_str());
    for (uint32_t b = 0; b < buffer_num; ++b) {
        // printf("Buffer %u:\n", b);
        for (uint32_t r = 0; r < depth; ++r) {
            // printf("Row %2u: ", r);
            for (uint32_t c = 0; c < width; ++c) {
                // printf("%d ", ready[b][r][c]);
            }
            // printf("\n");
    }}
    // printf("==========================================\n");
}

void
MatrixDmaDevice::DmaBuffer::printOccupied() const
{
    // printf("========= %s Occupied Dump =========\n", name.c_str());
    for (uint32_t b = 0; b < buffer_num; ++b) {
        // printf("Buffer %u:\n", b);
        for (uint32_t r = 0; r < depth; ++r) {
            // printf("Row %2u: ", r);
            for (uint32_t c = 0; c < width; ++c) {
                // printf("%d ", occupied[b][r][c]);
            }
            // printf("\n");
    }}
    // printf("============================================\n");
}

MatrixDmaDevice::DmaTransposeBuffer::DmaTransposeBuffer(uint32_t _buffer_num, uint32_t _width, uint32_t _depth, uint32_t cache_line_size, const std::string & _name)
    : DmaBuffer(_buffer_num, _width, _depth, cache_line_size, _name)
{
    transposeState.resize(_buffer_num, 0); // Initialize with 0
}
void 
MatrixDmaDevice::DmaTransposeBuffer::readCol(uint32_t num, uint32_t col, uint8_t* data_out, uint32_t size)
{
    assert(size == depth);
    assert(size > 0);
    for(uint32_t i = 0; i < depth; ++i) {
        data_out[i] = data[num][i][col];
        // printf("[wrp] Reading from %s at buffer %u, row %u, col %u, data %02x\n",
            //   name.c_str(), num, i, col, data_out[i]);
    }
}

} // namespace gem5