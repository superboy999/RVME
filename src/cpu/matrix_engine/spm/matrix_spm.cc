#include "base/random.hh"
#include "base/types.hh"
#include "sim/system.hh"
#include "sim/sim_object.hh"
#include "cpu/matrix_engine/spm/matrix_spm.hh"
#include "cpu/matrix_engine/SPMPacket.hh"
#include "cpu/matrix_engine/spm/matrix_lut_data.hh"
#include "mem/simple_mem.hh"
#include "debug/Drain.hh"
#include "debug/MatrixSPM.hh"
#include <vector>
#define CacheLineSize 64

using namespace std;

namespace gem5
{

MatrixSPM::MatrixSPM(const MatrixSPMParams &params) :
    memory::SimpleMemory(params),
    dmaDevice(params.dmaDevice),
    latency_write(params.latency_write),
    latency_write_var(params.latency_write_var),
    energy_read(params.energy_read),
    energy_write(params.energy_write),
    energy_overhead(params.energy_overhead),
    sramBanks(params.sram_banks),
    entriesPerBankA(params.entry_depthA),
    entriesPerBankB(params.entry_depthB),
    entriesPerBankC(params.entry_depthC),
    entryWidth(params.entry_width),
    rwportsPerBank(params.rw_ports_per_bank),
    rportsPerBank(params.r_ports_per_bank),
    wportsPerBank(params.w_ports_per_bank),
    lutEntries(params.lutEntries),
    RespEvent([this]{ processRespondEvent(); }, name() + ".RespEvent"),
    reqRecordIdx(0),
    lut_data()
{
    // Initialize banks
    for(int i = 0; i < sramBanks; i++) {
        banks.push_back(Bank(i, entriesPerBankA, entryWidth,
                            rwportsPerBank, rportsPerBank, wportsPerBank));
    }
    for(int i = 0; i < sramBanks; i++) {
        banks.push_back(Bank(i+sramBanks, entriesPerBankB, entryWidth,
                            rwportsPerBank, rportsPerBank, wportsPerBank));
    }
    for(int i = 0; i < sramBanks; i++) {
        banks.push_back(Bank(i+sramBanks*2, entriesPerBankC, entryWidth,
                            rwportsPerBank, rportsPerBank, wportsPerBank));
    }
    // Initialize spm_data
    spm_data.resize(sramBanks * 3);
    for(unsigned i = 0; i < sramBanks; i++) {
        spm_data[i].resize(entriesPerBankA);
        for (unsigned j = 0; j < entriesPerBankA; ++j) {
            spm_data[i][j].resize(entryWidth, 0);
        }
    }
    for(unsigned i = 0; i < sramBanks; ++i) {
        spm_data[i + sramBanks].resize(entriesPerBankB);
        for (unsigned j = 0; j < entriesPerBankB; ++j) {
            spm_data[i + sramBanks][j].resize(entryWidth, 0);
        }
    }
    for(unsigned i = 0; i < sramBanks; ++i) {
        spm_data[i + sramBanks * 2].resize(entriesPerBankC);
        for (unsigned j = 0; j < entriesPerBankC; ++j) {
            spm_data[i + sramBanks * 2][j].resize(entryWidth, 0);
        }
    }
    // Initialize SPM total size
    SPMTotalSize = sramBanks * (entriesPerBankA + entriesPerBankB + entriesPerBankC) * entryWidth;
    
    // printf("MatrixSPM created with %u banks in set A, %u entries per bank; "
    //     "%u banks in set B, %u entries per bank; %u banks in set C, %u entries per bank. "
    //     "Each bank has %u bytes entry size, %u r ports, %u w ports, %u rw ports "
    //     "and total SPM size is %u bytes.\n",
    //     sramBanks, entriesPerBankA, sramBanks, entriesPerBankB, sramBanks, entriesPerBankC,
    //     entryWidth, rportsPerBank, wportsPerBank, rwportsPerBank, SPMTotalSize);
    // printSpmData();

    lut_data.resize(lutEntries, 0);
    for (unsigned i = 0; i < INV_LUT_SIZE; ++i) {
        lut_data[i] = INV_LUT[i];
    }
    for (unsigned i = 0; i < 1024 - INV_LUT_SIZE; ++i) {
        lut_data[i + INV_LUT_SIZE] = 0;
    }
    unsigned offset = 1024;
    for (unsigned i = 0; i < FWD_LUT_SIZE; ++i) {
        lut_data[offset + i] = FWD_LUT[i];
    }
    for (unsigned i = 0; i < 256 - FWD_LUT_SIZE; ++i) {
        lut_data[offset + FWD_LUT_SIZE + i] = 0;
    } 
    last_access_lut_time = 0;
    // printLutData(0, 32);

    dmaDevice->set_spm_ptr(this);

    global_fence = 0;
    fence_threshold = 0;
}

MatrixSPM::~MatrixSPM() {}

void
MatrixSPM::init()
{
    memory::SimpleMemory::init();
}

void
MatrixSPM::regStats()
{
    using namespace statistics;

    AbstractMemory::regStats();

    System *system = (AbstractMemory::system());

    readEnergy
        .name(name() + ".energy_read")
        .desc("Total energy reading (pJ)")
        .precision(0)
        // .prereq(AbstractMemory::MemStats::numReads)
        .prereq(stats.numReads)
        .flags(total | nozero | nonan)
        ;
    for (int i = 0; i < system->maxRequestors(); i++) {
        readEnergy.subname(i, system->getRequestorName(i));
    }

    writeEnergy
        .name(name() + ".energy_write")
        .desc("Total energy writting (pJ)")
        .precision(0)
        // .prereq(AbstractMemory::MemStats::numWrites)
        .prereq(stats.numReads)
        .flags(total | nozero | nonan)
        ;
    for (int i = 0; i < system->maxRequestors(); i++) {
        writeEnergy.subname(i, system->getRequestorName(i));
    }

    overheadEnergy
        .name(name() + ".energy_overhead")
        .desc("Other energy (pJ)")
        .precision(0)
        // .prereq(AbstractMemory::numOther)
        .prereq(stats.numOther)
        .flags(total | nozero | nonan)
        ;
    for (int i = 0; i < system->maxRequestors(); i++) {
        overheadEnergy.subname(i, system->getRequestorName(i));
    }

    totalEnergy
        .name(name() + ".energy_total")
        .desc("Total energy (pJ)")
        .precision(0)
        .prereq(overheadEnergy)
        .flags(total | nozero | nonan)
        ;
    for (int i = 0; i < system->maxRequestors(); i++) {
        totalEnergy.subname(i, system->getRequestorName(i));
    }

    averageEnergy
        .name(name() + ".energy_average")
        .desc("Average energy (pJ)")
        .precision(0)
        .prereq(totalEnergy)
        .flags(total | nozero | nonan)
        ;
    for (int i = 0; i < system->maxRequestors(); i++) {
        averageEnergy.subname(i, system->getRequestorName(i));
    }

    readReqs
        .name(name() + ".readReqs")
        .desc("Number of read requests accepted");

    writeReqs
        .name(name() + ".writeReqs")
        .desc("Number of write requests accepted");

    readEntries
        .name(name() + ".readEntries")
        .desc("Number of DRAM read entries, "
              "including those serviced by the write queue");

    writeEntries
        .name(name() + ".writeEntries")
        .desc("Number of DRAM write entries, "
              "including those merged in the write queue");

    bytesReadSys
        .name(name() + ".bytesReadSys")
        .desc("Total read bytes from the system interface side");

    bytesWrittenSys
        .name(name() + ".bytesWrittenSys")
        .desc("Total written bytes from the system interface side");

    readPktSize
        .init(ceilLog2(entryWidth) + 1)
        .name(name() + ".readPktSize")
        .desc("Read request sizes (log2)");

    writePktSize
        .init(ceilLog2(entryWidth) + 1)
        .name(name() + ".writePktSize")
        .desc("Write request sizes (log2)");

    // Trying to implement a energy model...
    // readEnergy = AbstractMemory::numReads * energy_read;
    // writeEnergy = AbstractMemory::numWrites * energy_write;
    // overheadEnergy = AbstractMemory::numOther * energy_overhead;
    readEnergy = stats.numReads * energy_read;
    writeEnergy = stats.numWrites * energy_write;
    overheadEnergy = stats.numOther * energy_overhead;
    totalEnergy = readEnergy + writeEnergy + overheadEnergy;
    averageEnergy = (energy_overhead==0) ? totalEnergy / 2 : totalEnergy / 3 ;
}

void
MatrixSPM::set_matrixEngine_ptr(MatrixEngine* _matrix_engine)
{
    matrix_engine = _matrix_engine;
}

Tick
MatrixSPM::recvAtomic(PacketPtr pkt)
{
    access(pkt);
    return (pkt->isRead()) ? getLatency() : getWriteLatency();
}

Addr
MatrixSPM::encodeAddr(unsigned bankid, unsigned entryid, unsigned offset) const
{
    Addr addr = 0;
    if (bankid < sramBanks) {
        addr = entryid * sramBanks * entryWidth + bankid * entryWidth + offset;
    } else if (bankid < sramBanks * 2) {
        addr = entryid * sramBanks * entryWidth + (bankid - sramBanks) * entryWidth + offset
            + sramBanks * entriesPerBankA * entryWidth;
    } else {
        addr = entryid * sramBanks * entryWidth + (bankid - sramBanks * 2) * entryWidth + offset
            + sramBanks * entriesPerBankA * entryWidth + sramBanks * entriesPerBankB * entryWidth;
    }
    return addr;
}

MatrixSPM::BankEntry
MatrixSPM::decodeAddr(Addr addr) const
{
    assert(addr < SPMTotalSize);
    uint32_t offset   = addr % entryWidth;
    uint32_t entry_idx = addr / entryWidth;
    uint32_t bank_id  = entry_idx % sramBanks;
    uint32_t entry_id = entry_idx / sramBanks;
    uint32_t setAsize = sramBanks * entriesPerBankA * entryWidth;
    uint32_t setBsize = sramBanks * entriesPerBankB * entryWidth;
    uint8_t setid = 0;
    if (addr < setAsize) {
        // Set A
        assert(bank_id < sramBanks);
        assert(entry_id < entriesPerBankA); 
    } else if (addr < (setAsize + setBsize)) {
        // Set B
        assert(bank_id < sramBanks);
        assert(entry_id < entriesPerBankA + entriesPerBankB);
        setid = 1;
        bank_id += sramBanks;
        entry_id -= entriesPerBankA;
    } else {
        // Set C
        assert(bank_id < sramBanks);
        assert(entry_id < entriesPerBankA + entriesPerBankB + entriesPerBankC); 
        setid = 2;
        bank_id += sramBanks * 2;
        entry_id -= entriesPerBankA + entriesPerBankB; 
    }
    return BankEntry(setid, bank_id, entry_id, offset);
}

bool
MatrixSPM::addToBankQueue(PacketPtr pkt, unsigned int pktCount)
{
    assert(pkt->isWrite() || pkt->isRead());
    assert(pktCount != 0);
    // std::vector<BankPacket*> b_pkt_vec;
    Addr addr = pkt->getAddr();
    BurstHelper* burst_helper = NULL;
    SPMPacketPtr spm_pkt = dynamic_cast<SPMPacketPtr>(pkt);
    unsigned bytes_done = 0;
    unsigned setid_check;
    DPRINTF(MatrixSPM, "Request to addr %#llx, size %lld, will translate to %d "
                "spm requests with AccessMode %s\n", pkt->getAddr(), pkt->getSize(), pktCount,
                AccessModeToString(spm_pkt->accessMode));
    if(spm_pkt->accessMode == AccessMode::Continuous) {
        // 检查地址是否超出范围
        assert(addr + pkt->getSize() <= SPMTotalSize);
        for (int cnt = 0; cnt < pktCount; ++cnt) {
            unsigned size = std::min((addr | (entryWidth - 1)) + 1,
                            pkt->getAddr() + pkt->getSize()) - addr;          
            BankEntry bank_entry = decodeAddr(addr);
            BankPacket* b_pkt = new BankPacket(pkt, pkt->isRead(), bank_entry.setid, bank_entry.bankid,
                bank_entry.entryid, addr, size, bytes_done);
            DPRINTF(MatrixSPM, "addToBankQueue creat an BankPacket with setid %u, bankid %u, "
                "entryid %u, offset %u, size %llu\n", b_pkt->setid, b_pkt->bank,
                b_pkt->entry, bank_entry.offset, size);
            // 检查是否跨越set访问
            if(cnt == 0)
                setid_check = b_pkt->setid;
            else
                assert(setid_check == b_pkt->setid);         
            // Make the burst helper for split packets
            if (pktCount > 1 && burst_helper == NULL) {
                burst_helper = new BurstHelper(pktCount);
            }
            b_pkt->burstHelper = burst_helper;
            if (pkt->isRead()) {
                readPktSize[ceilLog2(size)]++;
                readEntries++;
            } else {    
                writePktSize[ceilLog2(size)]++;
                writeEntries++;
            }
            assert(b_pkt != nullptr);
            updateBankPacketReadyTime(b_pkt);
            DPRINTF(MatrixSPM, "Adding spm pkt to bank %u Queue with addr "
                "%#llx, size %d\n", b_pkt->bank, b_pkt->addr, b_pkt->size);
            banks[b_pkt->bank].Queue.push_back(b_pkt);
            if(RespEvent.scheduled()) {
                Tick scheduled_time = RespEvent.when();
                if(scheduled_time > b_pkt->readyTime) {
                    assert(b_pkt->readyTime >= curTick());
                    DPRINTF(MatrixSPM, "Rescheduling RespEvent at %llu\n",
                            b_pkt->readyTime);
                    reschedule(RespEvent, b_pkt->readyTime);
                } else {
                    DPRINTF(MatrixSPM, "RespEvent already scheduled at %llu, "
                            "not rescheduling\n", scheduled_time);
                }
            } else {
                DPRINTF(MatrixSPM, "Scheduling RespEvent at %llu\n",
                        b_pkt->readyTime);
                schedule(RespEvent, b_pkt->readyTime);
            }
            bytes_done += size;
            addr = (addr | (entryWidth - 1)) + 1;
        }
    } else if (spm_pkt->accessMode == AccessMode::Cyclic) {
        // 循环访问必须满足size是一个set中一个整bank row的整数倍
        // 即sramBanks*entryWidth
        assert (pkt->getSize() % (sramBanks * entryWidth) == 0);
        // assert (pktCount % sramBanks == 0);
        Addr set_base_addr = pkt->getAddr() - pkt->getAddr() % (sramBanks * entryWidth);
        Addr end_addr = set_base_addr + pkt->getSize();
        assert(end_addr <= SPMTotalSize);
        for (int cnt = 0; cnt < pktCount; ++cnt) {
            // unsigned size = std::min((addr | (entryWidth - 1)) + 1,
            //                 pkt->getAddr() + pkt->getSize()) - addr;
            unsigned size = std::min((addr | (entryWidth - 1)) + 1,
                            addr + pkt->getSize() - bytes_done) - addr;           
            BankEntry bank_entry = decodeAddr(addr);
            BankPacket* b_pkt = new BankPacket(pkt, pkt->isRead(), bank_entry.setid, bank_entry.bankid,
                bank_entry.entryid, addr, size, bytes_done);
            DPRINTF(MatrixSPM, "addToBankQueue creat an BankPacket with setid %u, bankid %u, "
                "entryid %u, offset %u, size %llu\n", b_pkt->setid, b_pkt->bank,
                b_pkt->entry, bank_entry.offset, size); 
            // Make the burst helper for split packets
            if (pktCount > 1 && burst_helper == NULL) {
                burst_helper = new BurstHelper(pktCount);
            }
            b_pkt->burstHelper = burst_helper;
            if (pkt->isRead()) {
                readPktSize[ceilLog2(size)]++;
                readEntries++;
            } else {    
                writePktSize[ceilLog2(size)]++;
                writeEntries++;
            }
            assert(b_pkt != nullptr);
            updateBankPacketReadyTime(b_pkt);
            DPRINTF(MatrixSPM, "Adding spm pkt to bank %u Queue with addr "
                "%#llx, size %d\n", b_pkt->bank, b_pkt->addr, b_pkt->size);
            banks[b_pkt->bank].Queue.push_back(b_pkt);
            if(RespEvent.scheduled()) {
                Tick scheduled_time = RespEvent.when();
                if(scheduled_time > b_pkt->readyTime) {
                    assert(b_pkt->readyTime >= curTick());
                    DPRINTF(MatrixSPM, "Rescheduling RespEvent at %llu\n",
                            b_pkt->readyTime);
                    reschedule(RespEvent, b_pkt->readyTime);
                } else {
                    DPRINTF(MatrixSPM, "RespEvent already scheduled at %llu, "
                            "not rescheduling\n", scheduled_time);
                }
            } else {
                DPRINTF(MatrixSPM, "Scheduling RespEvent at %llu\n",
                        b_pkt->readyTime);
                schedule(RespEvent, b_pkt->readyTime);
            }
            bytes_done += size;
            addr = (addr | (entryWidth - 1)) + 1;
            if (addr >= end_addr) {
                addr = set_base_addr;
            }
        }
    } else {
        panic("Unknown access mode %s", AccessModeToString(spm_pkt->accessMode));
    }
    return true;
}

bool
MatrixSPM::recvTimingReq(PacketPtr pkt)
{
    panic_if(pkt->cacheResponding(), "Should not see packets where cache "
        "is responding");

    panic_if(!(pkt->isRead() || pkt->isWrite()),
        "Should only see read and writes at memory controller, "
        "saw %s to %#llx\n", pkt->cmdString(), pkt->getAddr());

    panic_if(pkt->getAddr() >=  SPMTotalSize,
             "SPM access address %#llx out of range!", pkt->getAddr());

    unsigned size = pkt->getSize();
    unsigned offset = pkt->getAddr() & (entryWidth - 1);
    unsigned int b_pkt_count = divCeil(offset + size, entryWidth);

    if (pkt->isRead()) {
        assert(size != 0);
        addToBankQueue(pkt, b_pkt_count);
        readReqs++;
        bytesReadSys += size;
    } else if (pkt->isWrite()) {
        assert(size != 0);
        addToBankQueue(pkt, b_pkt_count);
        writeReqs++;
        bytesWrittenSys += size;
    } else {
        DPRINTF(MatrixSPM, "Neither read nor write, ignore timing\n");
        // neitherReadNorWrite++;
        Tick totLat = ((pkt->isRead()) ? getLatency() : 0) + ((pkt->isWrite()) ? getWriteLatency() : 0);      
        accessAndRespond(pkt, totLat);
    }
    return true;
}

// 处理逻辑是遍历每个bank的respQueue的b_pkt的readyTime
// 若小于等于当前时刻，则处理该b_pkt的响应
// 遍历完成后，在所有剩余pkt中寻找下一个最早的响应时间，重新调度RespEvent
void 
MatrixSPM::processRespondEvent()
{
    // 1. 遍历所有bank，处理所有readyTime <= curTick()的b_pkt
    for (auto& b : banks) {
        DPRINTF(MatrixSPM, "RespondEvent going into bank %d\n", b.bankid);
        for (auto it = b.Queue.begin(); it != b.Queue.end();) {
            assert(*it != nullptr);
            DPRINTF(MatrixSPM, "go to resp %d\n", 
                        std::distance(b.Queue.begin(), it));
            BankPacket* b_pkt = *it;
            if (b_pkt->readyTime <= curTick()) {
                DPRINTF(MatrixSPM, "Bank %d: A b_pkt has reached its readyTime in "
                        "entry %u\n", b_pkt->bank, b_pkt->entry);
                if (b_pkt->burstHelper) {
                    // it is a split packet
                    b_pkt->burstHelper->burstsServiced++;
                    bank_access(b_pkt);
                    if (b_pkt->burstHelper->burstsServiced == b_pkt->burstHelper->burstCount) {
                        Respond(b_pkt->pkt, 0);
                        delete b_pkt->burstHelper;
                        b_pkt->burstHelper = NULL;
                    }
                } else {
                    // it is not a split packet
                    accessAndRespond(b_pkt->pkt, 0);
                }
                it = b.Queue.erase(it);
                delete b_pkt;
            } else {
                ++it;
            }
        }
    }
    // 2. 在所有剩余pkt中寻找下一个最早的响应时间
    Tick min_ready = MaxTick;
    for (auto& b : banks) {
        if (!b.Queue.empty() && b.Queue.front()->readyTime < min_ready) {
            min_ready = b.Queue.front()->readyTime;
        }
    }
    // 3. 重新调度RespEvent
    if (min_ready != MaxTick) {
        DPRINTF(MatrixSPM, "Rescheduling RespEvent at %llu\n", min_ready);
        schedule(RespEvent, min_ready);
    }
    // printSpmData(0, 64);
}

void 
MatrixSPM::updateBankPacketReadyTime(BankPacket* b_pkt)
{
    assert(b_pkt != NULL);
    assert(b_pkt->size <= entryWidth);
    // 计算访问延迟
    // Tick access_latency = (b_pkt->isRead) ? getLatency() : getWriteLatency();
    b_pkt->readyTime = curTick() + clockPeriod();
}

void
MatrixSPM::forwardRecord(PacketPtr pkt)
{
    BankEntry bank_entry = decodeAddr(pkt->getAddr());
    for (auto it = reqRecordQueue.begin(); it != reqRecordQueue.end(); ) {
        // printf("it = %u\n", it);
        // printf("it->setId = %u, bank_entry.setid = %u\n", it->setId, bank_entry.setid);
        if (it->setId == bank_entry.setid) {
            // printf("remainingRows = %u, alreadyInAccess = %u\n", it->remainingRows, it->alreadyInAccess);
            if ((it->remainingRows == 0) && it->alreadyInAccess) {
                unsigned bank_base = it->setId * sramBanks;
                unsigned bank_end  = bank_base + sramBanks;
                DPRINTF(MatrixSPM, "Release %s ports in set %u: Bank %u to Bank %u\n",
                    it->isRead ? "read/rw" : "write/rw", it->setId, bank_base, bank_end-1);
                for (unsigned i = bank_base; i < bank_end; ++i) {
                    banks[i].releaseBusyPort(it->isRead);
                }
                DPRINTF(MatrixSPM, "Remove record %lu in reqRecordQueue with setId %u, reqId %u\n", it->recordIdx, it->setId, it->reqId);
                it = reqRecordQueue.erase(it);
            }
            return;
        } else {
            ++it;
        }
    }
}

void
MatrixSPM::Respond(PacketPtr pkt, Tick static_latency)
{
    bool needsResponse = pkt->needsResponse();

    forwardRecord(pkt);

    if (needsResponse) {
        pkt->makeResponse();
        assert(pkt->isResponse());
        // Tick response_time = curTick() + static_latency + pkt->headerDelay +
        //                      pkt->payloadDelay;
        // Here we reset the timing of the packet before sending it out.
        pkt->headerDelay = pkt->payloadDelay = 0;

        // queue the packet in the response queue to be sent out after
        // the static latency has passed

        // port.schedTimingResp(pkt, response_time);
        // matrix_engine->matrix_mmu->Callback(pkt);
        matrix_engine->Callback(pkt);
    }
}

void
MatrixSPM::accessAndRespond(PacketPtr pkt, Tick static_latency)
{
    DPRINTF(MatrixSPM, "Responding to Address %#llx ..\n",pkt->getAddr());

    bool needsResponse = pkt->needsResponse();
    // do the actual memory access which also turns the packet into a
    // response
    // access(pkt);
    spm_access(pkt);
    forwardRecord(pkt);
    // turn packet around to go back to requester if response expected
    if (needsResponse) {
        // access already turned the packet into a response
        assert(pkt->isResponse());
        // response_time consumes the static latency and is charged also
        // with headerDelay that takes into account the delay provided by
        // the xbar and also the payloadDelay that takes into account the
        // number of data beats.
        // Tick response_time = curTick() + static_latency + pkt->headerDelay +
        //                      pkt->payloadDelay;
        // Here we reset the timing of the packet before sending it out.
        pkt->headerDelay = pkt->payloadDelay = 0;

        // queue the packet in the response queue to be sent out after
        // the static latency has passed

        // port.schedTimingResp(pkt, response_time);
        // matrix_engine->matrix_mmu->Callback(pkt);
        matrix_engine->Callback(pkt);
    }
}

Tick
MatrixSPM::getWriteLatency() const
{
    return latency_write +
        (latency_write_var ? rng_wr->random<Tick>(0, latency_write_var) : 0);
        // (latency_write_var ? random_mt.random<Tick>(0, latency_write_var) : 0);
}

// 单次bank/entry访问
void MatrixSPM::bank_access(BankPacket* b_pkt)
{
    assert(b_pkt != nullptr);
    assert(b_pkt->bank < spm_data.size());
    assert(b_pkt->entry < spm_data[b_pkt->bank].size());
    assert(b_pkt->size <= entryWidth);
    unsigned offset = b_pkt->addr % entryWidth;
    unsigned bytes_this_entry = b_pkt->size;
    PacketPtr pkt = b_pkt->pkt;
    if (b_pkt->isRead) {
        // 从 SPM 读到 pkt
        std::memcpy(pkt->getPtr<uint8_t>() + b_pkt->start_bytes,
                    &spm_data[b_pkt->bank][b_pkt->entry][offset],
                    bytes_this_entry);
        // printf("SPM Read: Bank %u, Entry %u, Offset %u, Size %u, Bank Data: [",
        //         b_pkt->bank, b_pkt->entry, offset, bytes_this_entry);
        for (unsigned i = 0; i < bytes_this_entry; ++i) {
            // printf("%02x ", spm_data[b_pkt->bank][b_pkt->entry][offset + i]);
        }
        // printf("], Pkt Data: [");
        // for (unsigned i = 0; i < bytes_this_entry; ++i) {
        //     printf("%02x ", pkt->getPtr<uint8_t>()[b_pkt->start_bytes + i]);
        // }
        // printf("]\n");
    } else if (pkt->isWrite()) {
        // 从 pkt 写到 SPM
        std::memcpy(&spm_data[b_pkt->bank][b_pkt->entry][offset],
                    pkt->getPtr<uint8_t>() + b_pkt->start_bytes,
                    bytes_this_entry);
        // printf("SPM Write: Bank %u, Entry %u, Offset %u, Size %u, Pkt Data: [",
        //         b_pkt->bank, b_pkt->entry, offset, bytes_this_entry);
        // for (unsigned i = 0; i < bytes_this_entry; ++i) {
        //     printf("%02x ", pkt->getPtr<uint8_t>()[b_pkt->start_bytes + i]);
        // }
        // printf("], Bank Data: [");
        // for (unsigned i = 0; i < bytes_this_entry; ++i) {
        //     printf("%02x ", spm_data[b_pkt->bank][b_pkt->entry][offset + i]);
        // }
        // printf("]\n");
    } else {
        panic("SPM only supports read/write packets!");
    }
}

void 
MatrixSPM::spm_access(PacketPtr pkt)
{
    assert(pkt != nullptr);
    Addr addr = pkt->getAddr();
    unsigned size = pkt->getSize();
    unsigned bytes_done = 0;
    while (bytes_done < size) {
        // 计算当前 entry 的 bank、entry、offset
        BankEntry bank_entry = decodeAddr(addr);
        assert(bank_entry.bankid < spm_data.size());
        assert(bank_entry.entryid < spm_data[bank_entry.bankid].size());
        // 本 entry 可操作的最大字节数
        unsigned bytes_this_entry = std::min(entryWidth - bank_entry.offset, size - bytes_done);
        if (pkt->isRead()) {
            // 从 SPM 读到 pkt
            std::memcpy(pkt->getPtr<uint8_t>() + bytes_done,
                        &spm_data[bank_entry.bankid][bank_entry.entryid][bank_entry.offset],
                        bytes_this_entry);
        } else if (pkt->isWrite()) {
            // 从 pkt 写到 SPM
            std::memcpy(&spm_data[bank_entry.bankid][bank_entry.entryid][bank_entry.offset],
                        pkt->getPtr<uint8_t>() + bytes_done,
                        bytes_this_entry);
        } else {
            panic("SPM only supports read/write packets!");
        }
        addr += bytes_this_entry;
        bytes_done += bytes_this_entry;
    }
    if (pkt->needsResponse()) {
        pkt->makeResponse();
    }
}

bool MatrixSPM::isAvailable(uint8_t setid, uint8_t rows, bool isread, uint32_t fence, uint64_t reqId)
{
    DPRINTF(MatrixSPM, "Checking availability of SPM, setid=%u, rows=%u, isread=%d, fence=%u\n", setid, rows, isread, fence);
    if(fence > fence_threshold){
        DPRINTF(MatrixSPM, "Instruction is blocked by fence, now fence_threshold = %u, instruction fence = %u\n", fence_threshold, fence);
        return false;
    }
    assert(0 <= setid && setid <= 2);
    bool found_record_in_set_head = false;
    bool firstAccess = false;
    for (auto it = reqRecordQueue.begin(); it != reqRecordQueue.end(); ) {
        if (it->setId == setid) {
            if (it->reqId == reqId) {
                found_record_in_set_head = true;
                if(!it->alreadyInAccess) {
                    it->alreadyInAccess = true;
                    firstAccess = true;
                }
                it->remainingRows = rows;
                it->isRead = isread;
                DPRINTF(MatrixSPM, "Found reqId %u as the head instruction in set %u, remainingRows=%u, isRead=%d, firstAccess=%d\n", reqId, setid, it->remainingRows, it->isRead, firstAccess);
            }
            break;
        } else {
            ++it;
        }
    }
    if (!found_record_in_set_head) {
        DPRINTF(MatrixSPM, "reqId %u is not the head instruction in set %u\n", reqId, setid);
        return false;
    }

    unsigned bank_base = setid * sramBanks;
    unsigned bank_end  = bank_base + sramBanks;
    DPRINTF(MatrixSPM, "Checking banks available from Bank %u to Bank %u\n", bank_base, bank_end-1);
    bool havebusyport = false;
    // 先检查所有bank是否可用
    for (unsigned i = bank_base; i < bank_end; ++i) {
        const Bank& bank = banks[i];
        if (isread) {
            if (!bank.CanRead()) {
                DPRINTF(MatrixSPM, "Bank %u is not available for read\n", i);
                havebusyport = true;
            }
        } else {
            if (!bank.CanWrite()) {
                DPRINTF(MatrixSPM, "Bank %u is not available for write\n", i);
                havebusyport = true;
            }
        }
    }
    // if (havebusyport) {
    //     return false; // 有bank忙碌，不能继续
    // }

    if(firstAccess) {
        assert(!havebusyport);
        for (unsigned i = bank_base; i < bank_end; ++i) {
            Bank& bank = banks[i];
            if (isread) {
                bank.setReadBusy();
            } else {
                bank.setWriteBusy();
            }
        }
    }

    // assert(rows > 0);
    // reqRecordIdx++;
    // DPRINTF(MatrixSPM, "Adding reqRecord with idx %u, setid %u, rows %u, isread %d\n",
    //         reqRecordIdx, setid, rows, isread);
    // reqRecordQueue.emplace_back(isread, setid, rows, reqRecordIdx);
    return true;
}

bool
MatrixSPM::addToReqQueue(uint32_t reqId, uint8_t setid, bool isread)
{
    uint32_t recordId = reqRecordIdx++;
    DPRINTF(MatrixSPM, "Adding reqRecord with idx %u, setid %u into reqRecordQueue %u, isread = %d\n", reqId, setid, recordId, isread);
    reqRecordQueue.emplace_back(recordId, reqId, setid, isread);
    return true;
}

// bool
// MatrixSPM::sendTimingReadReq(Addr memAddr, Addr spmAddr, uint8_t size)
// {
//     DPRINTF(MatrixSPM, "Sending timing read request from SPM addr %#llx to mem addr %#llx, size = %u\n",
//             spmAddr, memAddr, size);
//     dmaDevice->addReq(MemCmd::ReadReq, spmAddr, memAddr, size);

//     return true;
// }

// bool
// MatrixSPM::sendTimingWriteReq(Addr memAddr, Addr spmAddr, uint8_t size)
// {
//     DPRINTF(MatrixSPM, "Sending timing write request from SPM addr %#llx to mem addr %#llx, size = %u\n",
//             spmAddr, memAddr, size);
//     dmaDevice->addReq(MemCmd::WriteReq, spmAddr, memAddr, size);
//     return true;
// }

// void
// MatrixSPM::printSpmData() const
// {
//     printf("========= SPM Data Dump =========\n");
//     for (unsigned bank = 0; bank < spm_data.size(); ++bank) {
//         printf("Bank %u:\n", bank);
//         for (unsigned entry = 0; entry < spm_data[bank].size(); ++entry) {
//             printf("  Entry %2u:  [0x%08llx] ", entry, (unsigned long long)encodeAddr(bank, entry, 0));
//             for (unsigned byte = 0; byte < spm_data[bank][entry].size(); ++byte) {
//                 printf("%02x ", spm_data[bank][entry][byte]);
//             }
//             printf("\n");
//         }
//     }
//     printf("=================================\n");
// }

void
MatrixSPM::printSpmData() const
{
    printf("========= SPM Data Dump =========\n");
    const unsigned nBanks = spm_data.size();
    for (unsigned bankBase = 0; bankBase < nBanks; bankBase += sramBanks) {
        const unsigned entries = spm_data[bankBase].size();
        for (unsigned entry = 0; entry < entries; ++entry) {
            printf("E%2u:", entry);
            for (unsigned b = bankBase; b < bankBase + sramBanks; ++b) {
                printf(" B%u:", b);
                for (auto byte : spm_data[b][entry])
                    printf("%02x", byte);
            }
            printf("\n");
        }
        printf("\n");
    }
    printf("=================================\n");
}

void
MatrixSPM::printSpmData(unsigned start_entry, unsigned num_entries) const
{
    printf("========= SPM Data Dump =========\n");
    const unsigned nBanks = spm_data.size();

    // 遍历 bank group（每个 group 包含 sramBanks 个 bank）
    for (unsigned bankBase = 0; bankBase < nBanks; bankBase += sramBanks) {
        const unsigned entries = spm_data[bankBase].size();
        if (entries == 0) {
            // 该 bank group 没数据，跳过
            continue;
        }

        // 计算实际打印范围：从 start_entry 到 end_entry-1
        if (start_entry >= entries) {
            // 起始行在这组中越界 -> 整组跳过
            continue;
        }
        unsigned max_print = std::min(entries, start_entry + num_entries);
        for (unsigned entry = start_entry; entry < max_print; ++entry) {
            printf("E%2u:", entry);
            for (unsigned b = bankBase; b < std::min(bankBase + sramBanks, nBanks); ++b) {
                printf(" B%u:", b);
                for (auto byte : spm_data[b][entry])
                    printf("%02x", byte);
            }
            printf("\n");
        }
        printf("\n");
    }
    printf("=================================\n");
}

void 
MatrixSPM::readSPM(Addr addr, uint8_t* data, uint32_t size)
{
    assert(size > 0);
    assert(addr + size <= SPMTotalSize);
    unsigned bytes_done = 0;
    while (bytes_done < size) {
        // 计算当前 entry 的 bank、entry、offset
        BankEntry bank_entry = decodeAddr(addr);
        assert(bank_entry.bankid < spm_data.size());
        assert(bank_entry.entryid < spm_data[bank_entry.bankid].size());
        // 本 entry 可操作的最大字节数
        unsigned bytes_this_entry = std::min(entryWidth - bank_entry.offset, size - bytes_done);
        // 从 SPM 读到 data
        std::memcpy(data + bytes_done,
                    &spm_data[bank_entry.bankid][bank_entry.entryid][bank_entry.offset],
                    bytes_this_entry);
        addr += bytes_this_entry;
        bytes_done += bytes_this_entry;
    }
}

void 
MatrixSPM::writeSPM(Addr spmAddr, const uint8_t* data, uint32_t size)
{
    assert(size > 0);
    assert(spmAddr + size <= SPMTotalSize);
    unsigned bytes_done = 0;
    while (bytes_done < size) {
        // 计算当前 entry 的 bank、entry、offset
        BankEntry bank_entry = decodeAddr(spmAddr + bytes_done);
        assert(bank_entry.bankid < spm_data.size());
        assert(bank_entry.entryid < spm_data[bank_entry.bankid].size());
        // 本 entry 可操作的最大字节数
        unsigned bytes_this_entry = std::min(entryWidth - bank_entry.offset, size - bytes_done);
        // 从 data 写到 SPM
        std::memcpy(&spm_data[bank_entry.bankid][bank_entry.entryid][bank_entry.offset],
                    data + bytes_done,
                    bytes_this_entry);
        // if(bytes_done == 0) {
        //     printf("SPM Write: Bank %u, Entry %u, Offset %u, Size %u, Pkt Data: ", bank_entry.bankid, bank_entry.entryid, bank_entry.offset);
        // }
        // printf("%02x ", spm_data[bank_entry.bankid][bank_entry.entryid][bank_entry.offset]);
        bytes_done += bytes_this_entry;
        // if(bytes_done >= size) 
        //     printf("\n");
    }
}

uint64_t
MatrixSPM::sendTimingReadReq(Addr memAddr, Addr spmAddr, uint8_t cachelines, std::shared_ptr<DmaReqState> shared_ptr)
{
    if(cachelines == 0) {
        cachelines = 32;
    }
    DPRINTF(MatrixSPM, "Sending timing read request from SPM addr %#llx to mem addr %#llx, size = %u\n",
        spmAddr, memAddr, cachelines * CacheLineSize);
    return dmaDevice->addReq(MemCmd::ReadReq, spmAddr, memAddr, cachelines * CacheLineSize, shared_ptr);
}

uint64_t
MatrixSPM::sendTimingWriteReq(Addr memAddr, Addr spmAddr, uint8_t cachelines, std::shared_ptr<DmaReqState> shared_ptr)
{ 
    if(cachelines == 0) {
        cachelines = 32;
    }
    DPRINTF(MatrixSPM, "Sending timing write request from SPM addr %#llx to mem addr %#llx, size = %u\n",
        spmAddr, memAddr, cachelines * CacheLineSize);
    return dmaDevice->addReq(MemCmd::WriteReq, spmAddr, memAddr, cachelines * CacheLineSize, shared_ptr);
}

uint64_t
MatrixSPM::sendMELoadSPM(std::shared_ptr<DmaReqState> shared_ptr, uint8_t setid)
{
    return dmaDevice->addReq(shared_ptr, setid, true);
}

uint64_t
MatrixSPM::sendMEStoreSPM(std::shared_ptr<DmaReqState> shared_ptr, uint8_t setid)
{
    return dmaDevice->addReq(shared_ptr, setid, false);
}

uint64_t
MatrixSPM::sendSync(std::shared_ptr<DmaReqState> shared_ptr)
{
    return dmaDevice->addSync(shared_ptr);
}

bool 
MatrixSPM::readLut(const uint32_t addrs[8], uint32_t out[8], uint32_t n, uint8_t mode)
{
    if (last_access_lut_time >= curTick()) {
        DPRINTF(MatrixSPM, "LUT busy: last=%llu now=%llu\n",
                last_access_lut_time, curTick());
        return false;
    }
    if (n == 0) return true;
    assert(n <= 8);
    assert(addrs != nullptr);
    assert(out != nullptr);
    assert(mode == 0 || mode == 1);

    constexpr uint32_t kInvEntries = 1024;   // 前半区大小
    constexpr uint32_t kFwdEntries = 256;    // 后半区大小

    uint32_t base  = (mode == 0) ? 0u : kInvEntries;       // mode=1 的物理起始 = 1024
    uint32_t limit = (mode == 0) ? kInvEntries : kFwdEntries;

    if (lut_data.size() < base + limit) {
        panic("LUT data size %u too small for mode %u "
              "(need at least %u)", lut_data.size(), mode, base + limit);
    }

    for (uint32_t i = 0; i < n; ++i) {
        uint32_t idx = addrs[i] % limit;
        if (idx >= limit) {
            DPRINTF(MatrixSPM, "addr OOB: mode=%u idx=%u limit=%u (i=%u)\n",
                    mode, idx, limit, i);
            return false;
        }
        out[i] = lut_data[base + idx];
    }

    last_access_lut_time = curTick();
    return true;
}

void
MatrixSPM::printLutData() const
{
    printf("========= LUT Data Dump =========\n");
    constexpr uint32_t kInvEntries = 1024;   // 前半区大小
    constexpr uint32_t kFwdEntries = 256;    // 后半区大小
    printf("Inverse LUT (mode=0):\n");
    for (uint32_t i = 0; i < kInvEntries; ++i) {
        printf("  [%4u] = 0x%08x\n", i, lut_data[i]);
    }
    printf("LayerNorm LUT (mode=1):\n");
    for (uint32_t i = 0; i < kFwdEntries; ++i) {
        printf("  [%4u] = 0x%08x\n", i, lut_data[kInvEntries + i]);
    }
    printf("=================================\n");
}

void
MatrixSPM::printLutData(unsigned start_entry, unsigned num_entries) const
{
    printf("========= LUT Data Dump =========\n");
    constexpr uint32_t kInvEntries = 1024;   // 前半区大小
    constexpr uint32_t kFwdEntries = 256;    // 后半区大小
    uint32_t max_print;
    // Inverse LUT
    printf("Inverse LUT (mode=0):\n");
    max_print = std::min(kInvEntries, start_entry + num_entries);
    for (uint32_t i = start_entry; i < max_print; ++i) {
        printf("  [%4u] = 0x%08x\n", i, lut_data[i]);
    }
    // LayerNorm LUT
    printf("LayerNorm LUT (mode=1):\n");
    max_print = std::min(kFwdEntries, start_entry + num_entries);
    for (uint32_t i = start_entry; i < max_print; ++i) {
        printf("  [%4u] = 0x%08x\n", i, lut_data[kInvEntries + i]);
    }
    printf("=================================\n");
}

} //namespace gem5