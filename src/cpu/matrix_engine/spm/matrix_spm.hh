
#ifndef __SCRATCHPAD_MEMORY_HH__
#define __SCRATCHPAD_MEMORY_HH__

#include <deque>
#include <memory>
#include "base/statistics.hh"
#include "base/random.hh"
#include "base/types.hh"
// #include "base/chunk_generator.hh"
// #include "dev/dma_device.hh"
#include "mem/simple_mem.hh"
#include "mem/port.hh"
#include "mem/request.hh"
#include "cpu/matrix_engine/matrix_engine.hh"
#include "cpu/matrix_engine/MatrixPacket.hh"
#include "cpu/matrix_engine/spm/matrix_dma.hh"
#include "params/SimpleMemory.hh"
#include "params/MatrixSPM.hh"
#include "debug/SPMBank.hh"
namespace gem5
{
class System;
class DmaReqState;

class MatrixSPM : public memory::SimpleMemory
{
  public:
      MatrixDmaDevice* dmaDevice;

      struct BankEntry
      {
        uint8_t setid; // 0: A, 1: B, 2: C
        uint8_t bankid;
        uint32_t entryid;
        uint32_t offset;
        BankEntry(uint8_t _setid, uint8_t _bankid, uint32_t _entryid, uint32_t _offset)
            : setid(_setid), bankid(_bankid), entryid(_entryid), offset(_offset) {}
      };
    
  private:
      // Latency if is a write request. If it is a read request,
      // latency from SimpleMemory is used (see implementation)
      const Tick latency_write;
      // Fudge factor added to the write latency.
      const Tick latency_write_var;
      // Fudge factor added to the write latency.
      const double energy_read;
      // Fudge factor added to the write latency.
      const double energy_write;
      // Fudge factor added to the write latency.
      const double energy_overhead;
      // Command energies
      statistics::Formula readEnergy;
      statistics::Formula writeEnergy;
      statistics::Formula overheadEnergy;
      statistics::Formula averageEnergy;
      statistics::Formula totalEnergy;
      // All statistics that the model needs to capture
      statistics::Scalar readReqs;
      statistics::Scalar writeReqs;
      statistics::Scalar bytesReadSys;
      statistics::Scalar bytesWrittenSys;
      statistics::Vector readPktSize;
      statistics::Vector writePktSize;
      statistics::Scalar readEntries;
      statistics::Scalar writeEntries;

  public:
      const uint32_t sramBanks;
      const uint32_t entriesPerBankA;
      const uint32_t entriesPerBankB;
      const uint32_t entriesPerBankC;
      const uint32_t entryWidth; // in bytes
      const uint32_t rwportsPerBank;
      const uint32_t rportsPerBank;
      const uint32_t wportsPerBank;
      // const uint32_t bankBufferSize;
      uint32_t SPMTotalSize;
      const uint32_t lutEntries;
      MatrixEngine* matrix_engine;

  private:
      mutable Random::RandomPtr rng_wr = Random::genRandom();

      void processRespondEvent();
      EventFunctionWrapper RespEvent;

      class BurstHelper {
        public:
          /** Number of SPM bursts requred for a system packet **/
          const unsigned int burstCount;
          /** Number of SPM bursts serviced so far for a system packet **/
          unsigned int burstsServiced;
          BurstHelper(unsigned int _burstCount)
              : burstCount(_burstCount), burstsServiced(0)
          { }
      };

      class BankPacket {
        public:
          /** When did request enter the controller */
          const Tick entryTime;
          /** This comes from the outside world */
          const PacketPtr pkt;
          const bool isRead;
          const Addr addr;
          const unsigned int size;
          const uint8_t setid; // 0: A, 1: B, 2: C         
          const uint8_t bank;
          const uint32_t entry;
          const uint32_t start_bytes; // 用来记录当前bankpkt在其父pkt中的位置
          /** true if this packet uses a read/write port */
          bool useRWPort;
          /** When will request leave the controller */
          Tick readyTime;
          /** The burst helper is used to track the number of bursts
           *  serviced for a system packet. */
          BurstHelper* burstHelper;
          BankPacket(PacketPtr _pkt, bool is_read, uint8_t _set, uint8_t _bank,
                    uint32_t _entry, Addr _addr, unsigned int _size, uint32_t _start_bytes)
              : entryTime(curTick()), pkt(_pkt), isRead(is_read),
                addr(_addr), size(_size), setid(_set), bank(_bank), entry(_entry),
                start_bytes(_start_bytes),
                useRWPort(false), readyTime(curTick()), burstHelper(NULL)
          { }
      };

      class Bank
      {
        public:
          const uint8_t bankid;
          const uint32_t entries;
          const uint32_t entryWidth; // in bytes
          const uint32_t rwports;
          const uint32_t rports;
          const uint32_t wports;
          uint32_t readBusyPort;
          uint32_t writeBusyPort;
          uint32_t rwBusyPort;
          std::deque<BankPacket*> Queue;
          bool CanRead() const {
              return (readBusyPort < rports) || 
                    (rwBusyPort < rwports);
          }
          bool CanWrite() const {
              return (writeBusyPort < wports) || 
                    (rwBusyPort < rwports);
          }
          void setReadBusy() {
            assert(CanRead());
            if (readBusyPort < rports) {
              readBusyPort++;
              // printf("[wrp] Bank %u: Set read busy port %u\n", bankid, readBusyPort);
            } else if (rwBusyPort < rwports) {
              rwBusyPort++;
              // printf("[wrp] Bank %u: Set rw busy port %u\n", bankid, rwBusyPort);
            } else {
              panic("Bank %u: Cannot set read busy, no free ports", bankid);
            }
          }
          void setWriteBusy() {
            assert(CanWrite());
            if (writeBusyPort < wports) {
              writeBusyPort++;
              // printf("[wrp] Bank %u: Set write busy port %u\n", bankid, writeBusyPort);
            } else if (rwBusyPort < rwports) {
              rwBusyPort++;
              // printf("[wrp] Bank %u: Set rw busy port %u\n", bankid, rwBusyPort);
            } else {
              panic("Bank %u: Cannot set write busy, no free ports", bankid);
            }
          }
          void releaseBusyPort(bool isRead) {
            if (isRead) {
              if (readBusyPort > 0) {
                assert(rwports == 0);
                readBusyPort--;
                // printf("[wrp] Bank %u: Released read busy port\n", bankid);
              } else if (rwBusyPort > 0) {
                rwBusyPort--;
                // printf("[wrp] Bank %u: Released rw busy port\n", bankid);
              } else {
                panic("Bank %u: Cannot release read busy, no busy ports", bankid);
              }
            } else {
              if (writeBusyPort > 0) {
                assert(rwports == 0);
                writeBusyPort--;
                // printf("[wrp] Bank %u: Released write busy port\n", bankid);
              } else if (rwBusyPort > 0) {
                rwBusyPort--;
                // printf("[wrp] Bank %u: Released rw busy port\n", bankid);
              } else {
                panic("Bank %u: Cannot release write busy, no busy ports", bankid);
              }
            }
          }
          Bank(uint32_t bankid, uint32_t entries, uint32_t entryWidth,
            uint32_t rwports, uint32_t rports, uint32_t wports) :
            bankid(bankid), entries(entries), entryWidth(entryWidth),
            rwports(rwports), rports(rports), wports(wports)
          {
              readBusyPort = 0;
              writeBusyPort = 0;
              rwBusyPort = 0;
          }
      };

      struct SPMReqRecord {
        uint32_t recordIdx = 0;
        uint64_t reqId = 0;
        uint8_t setId = 0;
        bool isRead = false;
        bool alreadyInAccess = false;
        uint8_t remainingRows = 0;
        SPMReqRecord(uint32_t _recordIdx, uint64_t _reqId, uint8_t _setId, bool _isRead)
            : recordIdx(_recordIdx), reqId(_reqId), setId(_setId), isRead(_isRead) {}
      };

      std::deque<SPMReqRecord> reqRecordQueue; // 用于记录ME请求
      uint32_t reqRecordIdx; // 用于记录当前reqRecordQueue的索引
      std::vector<Bank> banks;
      std::vector<std::vector<std::vector<uint8_t>>> spm_data; // [bank][entry][byte]
      std::vector<uint32_t> lut_data; // [entry]
      Tick last_access_lut_time;
      void forwardRecord(PacketPtr pkt);
      bool addToBankQueue(PacketPtr pkt, unsigned int pktCount);
      void updateBankPacketReadyTime(BankPacket* b_pkt);
      void Respond(PacketPtr pkt, Tick static_latency);
      void accessAndRespond(PacketPtr pkt, Tick static_latency);
      Tick getWriteLatency() const;
      void bank_access(BankPacket* b_pkt);
      void spm_access(PacketPtr pkt);

    public:
      BankEntry decodeAddr(Addr addr) const;
      Addr encodeAddr(unsigned bankid, unsigned entryid, unsigned offset) const;
      void readSPM(Addr spmAddr, uint8_t* data, uint32_t size);
      void writeSPM(Addr spmAddr, const uint8_t* data, uint32_t size);

  public:
    // Constructor and destructor
      MatrixSPM(const MatrixSPMParams &params);
      virtual ~MatrixSPM();
      void init();
      void regStats();
      void set_matrixEngine_ptr(MatrixEngine* _matrix_engine);
    // Methods for Matrix Engine access SPM
      bool isAvailable(uint8_t setid, uint8_t rows, bool isread, uint32_t fence, uint64_t reqId);
      Tick recvAtomic(PacketPtr pkt);
      bool recvTimingReq(PacketPtr pkt);
    // Methods for SPM access Memory
      // bool sendTimingReadReq(Addr memAddr, Addr spmAddr, uint8_t size);
      // bool sendTimingWriteReq(Addr memAddr, Addr spmAddr, uint8_t size);
    // Utils Functions
      void printSpmData() const;
      void printSpmData(unsigned start_entry, unsigned num_entries) const;
      void printLutData() const;
      void printLutData(unsigned start_entry, unsigned num_entries) const;
    // instruction Queue register
      uint64_t sendTimingReadReq(Addr memAddr, Addr spmAddr, uint8_t cachelines, std::shared_ptr<DmaReqState> shared_ptr);
      uint64_t sendTimingWriteReq(Addr memAddr, Addr spmAddr, uint8_t cachelines, std::shared_ptr<DmaReqState> shared_ptr);
      uint64_t sendMELoadSPM(std::shared_ptr<DmaReqState> shared_ptr, uint8_t setid);
      uint64_t sendMEStoreSPM(std::shared_ptr<DmaReqState> shared_ptr, uint8_t setid);
      uint64_t sendSync(std::shared_ptr<DmaReqState> shared_ptr);
    // Access Queue register
      bool addToReqQueue(uint32_t reqId, uint8_t setid, bool isread);
    // Access Lut
      bool readLut(const uint32_t addrs[8], uint32_t out[8], uint32_t n, uint8_t mode);

      uint64_t meminst_num = 0;
      bool meminst_Empty() const { return meminst_num == 0; }

      uint32_t global_fence; // 用于记录当前的全局fence值，每个meminst带有自己的fence值，每个sync push就会加1
      uint32_t fence_threshold; // 用于记录当前的fence阈值，只有sync pop才会加1
};
} //namespace gem5
#endif //__SCRATCHPAD_MEMORY_HH__
