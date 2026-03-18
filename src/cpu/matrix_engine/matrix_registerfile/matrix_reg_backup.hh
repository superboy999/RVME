#ifndef __CPU_MATRIX_REG_HH__
#define __CPU_MATRIX_REG_HH__

#include <cassert>
#include <cstdint>
#include <string>

#include "sim/clocked_object.hh"
// #include "mem/qport.hh"
// #include "mem/packet.hh"
// #include "base/types.hh"

namespace gem5
{
class MatrixRF : public ClockedObject
{
// public:
    // class MatrixRFPort : public QueuedResponsePort
    // {
    // public:
    //     MatrixRFPort(const std::string& name, MatrixRFPort& matrix_reg);
    // protected:
    //     bool recvTimingReq(PacketPtr pkt) override;
    //     /** A normal packet queue used to store responses. */
    //     RespPacketQueue queue;
    //     MatrixRF matrix_reg;
    // };

public:
    MatrixRF(const MatrixRFParams &params);
    ~MatrixRF();

    // Port& getPort(const std::name& name, PortID& idx = InvalidPortID) override;

    // will be called by the the port when receive the requestport's packet
    bool handleTimingReq(PacketPtr pkt, MatrixRFPort* port);
    void regStats() override;

    uint64_t get_size();
private:
    const uint64_t phy_reg_num;
    const uint64_t RLEN;
    const uint64_t lineSize;
    const uint64_t numBanks; //default is 4, because worst case is four lane write to the reg simultaneously


} 

} // namespace gem5


#endif