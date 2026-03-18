#ifndef __CPU_SPM_PACKET_HH__
#define __CPU_SPM_PACKET_HH__

#include <cstdint>
#include <functional>

#include "mem/packet.hh"
#include "mem/request.hh"

namespace gem5{
enum class AccessMode {
      Continuous,
      Cyclic
};

inline const char* AccessModeToString(AccessMode mode) {
    switch (mode) {
        case AccessMode::Continuous: return "Continuous";
        case AccessMode::Cyclic:     return "Cyclic";
        default:                     return "Unknown";
    }
}

class SPMPacket;
typedef SPMPacket* SPMPacketPtr;

class SPMPacket : public Packet
{
public:
    SPMPacket(RequestPtr req, MemCmd cmd, uint8_t _setid,
              uint32_t _bankid, uint32_t _entryid, AccessMode _accessMode = AccessMode::Continuous,
              std::function<void(uint8_t* data, uint8_t size)> _readCallback = nullptr)
        : Packet(req, cmd), accessMode(_accessMode), setid(_setid),
          bankid(_bankid), entryid(_entryid), finish(false),
          dst_reg(0), readCallback(_readCallback), writeCallback(nullptr) {}
    SPMPacket(RequestPtr req, MemCmd cmd, uint8_t _setid,
              uint32_t _bankid, uint32_t _entryid, AccessMode _accessMode = AccessMode::Continuous,
              std::function<void(void)> _writeCallback = nullptr)
        : Packet(req, cmd), accessMode(_accessMode), setid(_setid),
          bankid(_bankid), entryid(_entryid), finish(false),
          dst_reg(0), readCallback(nullptr), writeCallback(_writeCallback) {}
    SPMPacket(RequestPtr req, MemCmd cmd, AccessMode _accessMode,
              std::function<void(uint8_t* data, uint8_t size)> _readCallback)
        : Packet(req, cmd), accessMode(_accessMode), setid(0),
          bankid(0), entryid(0), finish(false),
          dst_reg(0), readCallback(_readCallback), writeCallback(nullptr) {}
    SPMPacket(RequestPtr req, MemCmd cmd, AccessMode _accessMode,
              std::function<void(void)> _writeCallback)
        : Packet(req, cmd), accessMode(_accessMode), setid(0),
          bankid(0), entryid(0), finish(false),
          dst_reg(0), readCallback(nullptr), writeCallback(_writeCallback) {}
    SPMPacket(RequestPtr req, MemCmd cmd, AccessMode _accessMode = AccessMode::Continuous)
        : Packet(req, cmd), accessMode(_accessMode), setid(0),
          bankid(0), entryid(0), finish(false),
          dst_reg(0), readCallback(nullptr), writeCallback(nullptr) {}
    ~SPMPacket() {};
public:
    AccessMode accessMode;
    // 可以直接指定起始setid、bankid、entryid，然后根据accessMode进行操作
    uint8_t setid;
    uint32_t bankid;
    uint32_t entryid;
    bool finish;
    uint8_t dst_reg;
    std::function<void(uint8_t* data, uint8_t size)> readCallback;
    std::function<void(void)> writeCallback;
};
}

#endif