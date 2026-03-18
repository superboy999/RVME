#ifndef __CPU_MATRIX_DMA_HH__
#define __CPU_MATRIX_DMA_HH__

#include <string>
#include "base/statistics.hh"
#include "base/types.hh"
#include "cpu/matrix_engine/spm/matrix_spm.hh"
#include "mem/port.hh"
#include "mem/packet.hh"
#include "mem/request.hh"
#include "sim/ticked_object.hh"
#include "params/MatrixDmaDevice.hh"
#include "arch/generic/mmu.hh"
#include "sim/system.hh"

namespace gem5
{
class DmaReqState
{
  public:
    const MemCmd cmd;
    const Addr spmAddr;
    const Addr memAddr;
    uint32_t totBytes;
    uint64_t reqId;
    bool needtranspose;
    bool isExecuted = false; // 为了使得外部告知spm此条指令已经完成（如mlae8.spm指令）
    bool isDMA;
    bool isFence;
    uint32_t fence;
    Addr bufferAddr;
    uint32_t issuedBytes = 0;
    uint32_t readyBytes = 0;
    uint32_t spmAccessBytes = 0;
    uint32_t finishedBytes = 0;
    ThreadContext& tc;
    
  DmaReqState(MemCmd _cmd, Addr _spmAddr, Addr _memAddr, uint32_t _tb, uint64_t _reqId, ThreadContext &_tc)
      : cmd(_cmd), spmAddr(_spmAddr), memAddr(_memAddr), totBytes(_tb), reqId(_reqId), tc(_tc)
  {}
  DmaReqState(MemCmd _cmd, Addr _spmAddr, Addr _memAddr, uint32_t _tb, uint64_t _reqId, bool _needtranspose, ThreadContext &_tc)
      : cmd(_cmd), spmAddr(_spmAddr), memAddr(_memAddr), totBytes(_tb), reqId(_reqId), needtranspose(_needtranspose), tc(_tc)
  {}
  DmaReqState(MemCmd _cmd, Addr _spmAddr, Addr _memAddr, uint32_t _tb, ThreadContext &_tc)
      : cmd(_cmd), spmAddr(_spmAddr), memAddr(_memAddr), totBytes(_tb), tc(_tc)
  {}
  DmaReqState(bool isDMA, bool isFence, ThreadContext &_tc)
      : cmd(MemCmd::MemSyncReq), spmAddr(0), memAddr(0), totBytes(0), isDMA(isDMA), isFence(isFence), tc(_tc)
  {}
};

class MatrixDmaDevice : public TickedObject
{
  public:
    MatrixDmaDevice(const MatrixDmaDeviceParams &params);
    void regStats() override;
    void evaluate() override;
    ~MatrixDmaDevice();
    Port& getPort(const std::string &if_name, PortID idx=InvalidPortID) override;
    RequestorID MatrixSPMRequestorId;

  public:
    class DmaPacket;
    typedef DmaPacket* DmaPacketPtr;
    class DmaPacket : public Packet
    {
      public:
        bool needtranspose;
        Addr bufferAddr;
        uint64_t reqId;
        DmaPacket(RequestPtr req, MemCmd cmd, bool needtranspose,
                  Addr _bufferAddr, uint64_t _reqId)
            : Packet(req, cmd), needtranspose(needtranspose),
              bufferAddr(_bufferAddr), reqId(_reqId) {}
        ~DmaPacket() {};
    };
    struct DmaSenderState : public Packet::SenderState {
        uint64_t reqId;
        bool     needTranspose;
        Addr     bufferAddr;
        unsigned size;
        DmaSenderState(uint64_t id, bool nt, Addr buf, unsigned sz)
          : reqId(id), needTranspose(nt), bufferAddr(buf), size(sz) {}
    };
    class DmaPort : public RequestPort
    {
      private:
        MatrixDmaDevice& device;
        bool recvTimingResp(PacketPtr pkt) override;
        void recvReqRetry() override;
      public:
        DmaPort(const std::string& _name, MatrixDmaDevice& _device, uint32_t _cache_line_size);
        ~DmaPort();
        PacketPtr inRetry = nullptr;
        // DmaPacketPtr inRetry = nullptr;
        bool retryiscalled = false;
        uint32_t cache_line_size;
    };
    struct DmaBufferEntry {
      uint32_t buffer_num;
      uint32_t row;
      uint32_t col;
      DmaBufferEntry(uint32_t _buffer_num, uint32_t _row, uint32_t _col)
          : buffer_num(_buffer_num), row(_row), col(_col) {}
    };
    class DmaBuffer
    {
      public:
        const uint32_t buffer_num;
        const uint32_t width;
        const uint32_t depth;
        const uint32_t cache_line_size;
        uint32_t totalSize;
        const std::string name;
      protected:
        std::vector<std::vector<std::vector<uint8_t>>> data; // [num][row][col]
        std::vector<std::vector<std::vector<bool>>> ready; // [num][row][col]
        std::vector<std::vector<std::vector<bool>>> occupied; // [num][row][col]  
      public:
        DmaBuffer(uint32_t _buffer_num, uint32_t _width, uint32_t _depth, uint32_t cache_line_size, const std::string & _name = "DmaBuffer");
        DmaBufferEntry decodeAddr(Addr addr) const;
        void read(Addr addr, uint8_t* data_out, uint32_t size) const;
        void write(Addr addr, const uint8_t* data_in, uint32_t size);
        void setReady(Addr addr, uint32_t size);
        void resetReady(Addr addr, uint32_t size);
        void setOccupied(Addr addr, uint32_t size);
        void resetOccupied(Addr addr, uint32_t size);
        bool isFree(Addr addr, uint32_t size) const;
        bool isReady(Addr addr, uint32_t size) const;
        bool findFree(uint32_t size) const;
        Addr getFreeAddr() const;
        void printBuffer() const;
        void printData() const;
        void printReady() const;
        void printOccupied() const;
        const std::string &getName() const { return name; }
    };
    class DmaTransposeBuffer : public DmaBuffer
    {
      public:
        std::vector<uint8_t> transposeState; // [num]
      public:
        DmaTransposeBuffer(uint32_t _buffer_num, uint32_t _width, uint32_t _depth, uint32_t cache_line_size, const std::string & _name = "DmaTransposeBuffer");
        void readCol(uint32_t num, uint32_t col, uint8_t* data_out, uint32_t size);
    };
    struct dmaAccessSpmEntry {
      bool isread;
      bool istranspose;
      Addr spm_addr;
      Addr buffer_addr;
      uint32_t size;
      // uint8_t bytes_done;
      // Tick issueTime;
      std::shared_ptr<DmaReqState> dmaReqState;
      uint32_t transposebuffer_num = 0;
      uint32_t transposebuffer_col = 0;
      dmaAccessSpmEntry(bool _isread, bool _istranspose, Addr _spm_addr, Addr _buffer_addr, uint32_t _size, std::shared_ptr<DmaReqState> dmaReqState = nullptr)
          : isread(_isread), istranspose(_istranspose), spm_addr(_spm_addr), buffer_addr(_buffer_addr),
            size(_size), dmaReqState(std::move(dmaReqState)) {}
      dmaAccessSpmEntry(bool _isread, bool _istranspose, Addr _spm_addr, Addr _buffer_addr, uint32_t _size, uint32_t _transposebuffer_num, uint32_t _transposebuffer_col, std::shared_ptr<DmaReqState> dmaReqState = nullptr)
          : isread(_isread), istranspose(_istranspose), spm_addr(_spm_addr), buffer_addr(_buffer_addr),
            size(_size), transposebuffer_num(_transposebuffer_num), transposebuffer_col(_transposebuffer_col), dmaReqState(std::move(dmaReqState)) {}
    };
    class DMA_Translation : public BaseMMU::Translation
    {
      public:
          DMA_Translation() : event(this, true) {}
          ~DMA_Translation() {}

          void markDelayed() override { panic("DMA Translation::markDelayed not implemented"); }

          void finish(const Fault &_fault,const RequestPtr &_req,
              ThreadContext *_tc, BaseMMU::Mode _mode) { fault = _fault; }
          void finish(const Fault _fault, uint64_t latency) { fault = _fault; }
          std::string name() { return "DMA_Translation"; }
      private:
          void translated() {}
          EventWrapper<DMA_Translation,&DMA_Translation::translated> event;
      public:
          Fault fault;
    };

  private:
    const uint32_t dma_buffer_width;
    const uint32_t dma_buffer_depth;
    const uint32_t dma_transpose_buffer_num;
    const uint32_t dma_transpose_buffer_width;
    const uint32_t dma_transpose_buffer_depth;
    const uint32_t cache_line_size;
    DmaPort dmaPort;
    MatrixSPM *matrix_spm;
    uint64_t dmaReqId;
    std::vector<std::deque<dmaAccessSpmEntry>> dmaAccessSpmQueue;
    
  protected:
    // std::deque<DmaReqState*> dmaReqQueue;
    std::deque<std::shared_ptr<DmaReqState>> dmaReqQueue;
    DmaBuffer dma_buffer;
    DmaTransposeBuffer dma_transpose_buffer;

  private:
    void accessSPM();
    void Issue();
    void scheduleAccessSPM();

  public:
    void set_spm_ptr(MatrixSPM *spm);
    // void addReq(MemCmd cmd, Addr spmAddr, Addr memAddr, uint32_t totBytes);
    uint64_t addReq(MemCmd cmd, Addr spmAddr, Addr memAddr, uint32_t totBytes, std::shared_ptr<DmaReqState> shared_ptr); // add dma load\store
    uint64_t addReq(std::shared_ptr<DmaReqState> shared_ptr, uint8_t setid, bool isread); // add spm load\store
    uint64_t addSync(std::shared_ptr<DmaReqState> shared_ptr); // add fence
    void startTicking(); //will used by upper module
    void stopTicking(); //可以在evaluate里面关掉也可以在外面关掉

    // Statistics
    statistics::Scalar dma_read_bytes;
    statistics::Scalar dma_write_bytes;
    statistics::Scalar dma_read_req;
    statistics::Scalar dma_write_req;
};
} // namespace gem5

#endif 