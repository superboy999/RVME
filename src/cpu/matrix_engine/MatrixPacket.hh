/*
 * @Author: superboy
 * @Date: 2024-06-26 16:57:46
 * @LastEditTime: 2024-07-28 22:20:42
 * @LastEditors: superboy
 * @Description: 
 * @FilePath: /gem5-rvm/src/cpu/matrix_engine/MatrixPacket.hh
 * 
 */
#ifndef __CPU_MATRIX_PACKET_HH__
#define __CPU_MATRIX_PACKET_HH__

#include <cstdint>
#include <functional>

#include "mem/packet.hh"
#include "mem/request.hh"

namespace gem5{
class MatrixPacket;
typedef MatrixPacket* MatrixPacketPtr;
class MatrixPacket : public Packet
{
public:
    MatrixPacket(RequestPtr req, MemCmd cmd, uint32_t req_id,uint8_t channel, uint8_t _dst_reg, std::function<void(uint8_t*data, uint8_t size)> readCallback):
        Packet(req, cmd), reqId(req_id), channel(channel), finish(false), dst_reg(_dst_reg), readCallback(readCallback) {}
    MatrixPacket(RequestPtr req, MemCmd cmd, uint32_t req_id, std::function<void(uint8_t*data, uint8_t size)> readCallback) :
        Packet(req, cmd), reqId(req_id), channel(0), finish(false),readCallback(readCallback) {}
    MatrixPacket(RequestPtr req, MemCmd cmd, uint32_t req_id,uint8_t channel, std::function<void(void)> writeCallback):
        Packet(req, cmd), reqId(req_id), channel(channel), finish(false), writeCallback(writeCallback) {}
    MatrixPacket(RequestPtr req, MemCmd cmd, uint32_t req_id, std::function<void(void)> writeCallback) :
        Packet(req, cmd), reqId(req_id), channel(0), finish(false),writeCallback(writeCallback) {}
    MatrixPacket(RequestPtr req, MemCmd cmd, uint32_t req_id,uint8_t channel):
        Packet(req, cmd), reqId(req_id), channel(channel), finish(false){}
    MatrixPacket(RequestPtr req, MemCmd cmd, uint32_t req_id) :
        Packet(req, cmd), reqId(req_id), channel(0), finish(false){}
    ~MatrixPacket() {};
    const uint32_t reqId;
    const uint8_t channel;
    bool finish;
    uint8_t dst_reg;
public:
    std::function<void(uint8_t* data, uint8_t size)> readCallback;
    std::function<void(void)> writeCallback;
};

}

#endif //__CPU_MATRIX_PACKET_HH__