#ifndef __CPU_MATRIX_MMU__
#define __CPU_MATRIX_MMU__

#include <cstdint>
#include <functional>
#include <queue>
#include <cassert>
#include <cstdint>
#include <string>
#include <deque>
#include <vector>
#include <memory>

#include "base/statistics.hh"
#include "cpu/matrix_engine/matrix_engine.hh"
#include "cpu/matrix_engine/MatrixPacket.hh"
#include "params/MatrixMMU.hh"
#include "sim/clocked_object.hh"
#include "mem/packet.hh"
#include "mem/port.hh"
#include "base/types.hh"
// #include "arch/generic/tlb.hh"
#include "arch/generic/mmu.hh"
#include "cpu/thread_context.hh"
// #include "cpu/o3/thread_context.hh"
#include "cpu/translation.hh"
#include "mem/request.hh"
#include "sim/process.hh"
#include "cpu/o3/cpu.hh"
#include "cpu/base.hh"
#include "sim/system.hh"
namespace gem5
{
class MatrixEngine;
struct MatrixMMUParams;
class MatrixPacket;

namespace o3{
    class CPU;
}//namespace o3
class MatrixMMU : public ClockedObject
{
public:
    MatrixMMU(const MatrixMMUParams &params);
    ~MatrixMMU();

    //rewrite the function inhereted from ClockedObject
    // void evaluate() override;
    void regStats() override;
    void set_cpu_ptr(gem5::o3::CPU* _o3cpu);
    void set_matrixEngine_ptr(MatrixEngine* _matrix_engine);
    // void configsize();
    class MatrixMemPort : public RequestPort
    {
    public:
        MatrixMemPort(const std::string& name,  MatrixMMU* owner, uint8_t port_id);
        ~MatrixMemPort();

        bool recvTimingResp(PacketPtr pkt) override;
        void recvReqRetry() override;

        bool startTranslation(Addr addr, uint8_t* data, uint32_t size, BaseMMU::Mode mode, ThreadContext *tc, uint32_t req_id, uint8_t channel, uint8_t port_id);
        bool sendTimingReadReq(Addr addr, uint32_t size, ThreadContext *tc, uint32_t req_id, uint8_t channel, uint8_t port_id);
        bool sendTimingWriteReq(Addr addr, uint8_t *data, uint32_t size, ThreadContext *tc, uint32_t req_id, uint8_t channel, uint8_t port_id);

        //For those failed data-transition! Vector size = num_channel
        std::vector<std::vector< std::deque<PacketPtr> >> laCachePktQs; //laCachePktQs[port][channel]
        MatrixMMU* owner;        
        const uint32_t port_id;    

        class MMU_Translation : public BaseMMU::Translation
        {
            public:
                MMU_Translation(MatrixMMU *owner);
                ~MMU_Translation();

                void markDelayed() override;

                void finish(const Fault &_fault,const RequestPtr &_req,
                    ThreadContext *_tc, BaseMMU::Mode _mode) ;
                void finish(const Fault _fault, uint64_t latency);
                std::string name();
            private:
                void translated();
                EventWrapper<MMU_Translation,&MMU_Translation::translated> event;
                MatrixMMU *owner;
            public:
                Fault fault;
        };
    };

public:
    //used to identify ports uniquely to whole memory system
    // RequestorID MatrixCacheRequestorId;
    // MatrixMemPort matrixmem_port;
    std::vector<RequestorID> MatrixCacheRequestorIds;
    std::vector<MatrixMemPort> matrix_mem_ports;
    
    Port& getPort(const std::string& if_name, PortID idx = InvalidPortID) override;
    RequestPort &getMatrixMemPort(PortID idx) {return matrix_mem_ports[idx];} 

    void recvTimingResp(MatrixPacketPtr pkt);

    bool writeMatrixMem(Addr addr, uint8_t* data, uint32_t size, ThreadContext *tc, uint8_t port_id, std::function<void(void)> writeCallback);
    bool readMatrixMem(Addr addr, uint32_t size, ThreadContext *tc, uint8_t port_id, uint8_t dst_reg, std::function<void(uint8_t* data, uint8_t size)> readCallback);

    //fields for keeping track of outstanding requests for reordering
    uint64_t uniqueReqId;
    std::vector<std::deque<MatrixPacketPtr>> matrix_PendingReqQs;// one pending queue per port(two channel)

    bool isFull();

private:

    // cwq I hope this value will be saved in two space, one is here and the other one will be inside the Matrix RegisterFile.
    // But they should be synchronize at same time when the configure instruction come into engine.
    uint8_t matrix_sizeM; // eight logic register are all the same controlled by one Msize Register
    uint8_t matrix_sizeN; 
    uint16_t matrix_sizeK;
private:
    // statistics::Scalar 


    //python configuration
    uint8_t num_matrix_mem_ports;
    uint32_t pending_depth;
    uint8_t w_channel;
    uint8_t r_channel;
private:
    gem5::o3::CPU *o3cpu;
    MatrixEngine* matrix_engine;
};

}//namespace gem5

#endif //__CPU_MATRIX_MMU__ 