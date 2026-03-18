/*
 * Copyright (c) 2015. Markos Horro
 * All rights reserved
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met: redistributions of source code must retain the above copyright
 * notice, this list of conditions and the following disclaimer;
 * redistributions in binary form must reproduce the above copyright
 * notice, this list of conditions and the following disclaimer in the
 * documentation and/or other materials provided with the distribution;
 * neither the name of the copyright holders nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * Authors: Markos Horro
 *
 */

#include "base/random.hh"
#include "base/types.hh"
#include "sim/system.hh"
#include "sim/sim_object.hh"
#include "mem/spm_mem.hh"
#include "mem/simple_mem.hh"
#include "debug/Drain.hh"
#include "debug/SPM.hh"

using namespace std;
 
namespace gem5
{
namespace memory
{

ScratchpadMemory::ScratchpadMemory(const ScratchpadMemoryParams &params) :
SimpleMemory(params),
latency_write(params.latency_write), latency_write_var(params.latency_write_var),
energy_read(params.energy_read), energy_write(params.energy_write),
energy_overhead(params.energy_overhead)
{
}

ScratchpadMemory::~ScratchpadMemory() {}

void
ScratchpadMemory::init()
{
    SimpleMemory::init();
}

void
ScratchpadMemory::regStats()
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

Tick
ScratchpadMemory::recvAtomic(PacketPtr pkt)
{
    access(pkt);
    return (pkt->isRead()) ? getLatency() : getWriteLatency();
}

bool
ScratchpadMemory::recvTimingReq(PacketPtr pkt)
{
    // printf("into spm recvTimingReq\n");
    // // fixme: by wrp
    // /// @todo temporary hack to deal with memory corruption issues until
    // /// 4-phase transactions are complete
    // // for (int x = 0; x < pendingDelete.size(); x++)
    // //     delete pendingDelete[x];
    // // pendingDelete.clear();
    
    
    // // if (pkt->memInhibitAsserted()) {
    // //     // snooper will supply based on copy of packet
    // //     // still target's responsibility to delete packet
    // //     pendingDelete.push_back(pkt);
    // //     return true;
    // // }

    // // we should never get a new request after committing to retry the
    // // current one, the bus violates the rule as it simply sends a
    // // retry to the next one waiting on the retry list, so simply
    // // ignore it
    // if (retryReq)
    //     return false;

    // // if we are busy with a read or write, remember that we have to
    // // retry
    // if (isBusy) {
    //     retryReq = true;
    //     return false;
    // }

    // pendingDelete.reset(pkt);
    // // @todo someone should pay for this
    // pkt->headerDelay = pkt->payloadDelay = 0;

    // // update the release time according to the bandwidth limit, and
    // // do so with respect to the time it takes to finish this request
    // // rather than long term as it is the short term data rate that is
    // // limited for any real memory

    // // only look at reads and writes when determining if we are busy,
    // // and for how long, as it is not clear what to regulate for the
    // // other types of commands
    // if (pkt->isRead() || pkt->isWrite()) {
    //     // calculate an appropriate tick to release to not exceed
    //     // the bandwidth limit
    //     Tick duration = pkt->getSize() * bandwidth;
    //     DPRINTF(SPM, "SPM duration = %llu\n", duration);

    //     // only consider ourselves busy if there is any need to wait
    //     // to avoid extra events being scheduled for (infinitely) fast
    //     // memories
    //     if (duration != 0) {
    //         schedule(releaseEvent, curTick() + duration);
    //         isBusy = true;
    //     }
    // }

    // // go ahead and deal with the packet and put the response in the
    // // queue if there is one
    // bool needsResponse = pkt->needsResponse();
    // recvAtomic(pkt);
    // // turn packet around to go back to requester if response expected
    // if (needsResponse) {
    //     // recvAtomic() should already have turned packet into
    //     // atomic response
    //     assert(pkt->isResponse());
    //     // to keep things simple (and in order), we put the packet at
    //     // the end even if the latency suggests it should be sent
    //     // before the packet(s) before it.

    //     // Difference between read/write latencies
    //     Tick totLat = ((pkt->isRead()) ? getLatency() : 0) + ((pkt->isWrite()) ? getWriteLatency() : 0);
    //     DPRINTF(SPM, "SPM totLat = %llu\n", totLat);
    //     packetQueue.push_back(DeferredPacket(pkt, curTick() + totLat));
    //     if (!retryResp && !dequeueEvent.scheduled())
    //         schedule(dequeueEvent, packetQueue.back().tick);
    // } else {
    //     // fixme: by wrp
    //     // pendingDelete.push_back(pkt);
        
    // }

    // printf("quit spm recvTimingReq\n");
    // return true;


    printf("into spm recvTimingReq\n");
    panic_if(pkt->cacheResponding(), "Should not see packets where cache "
             "is responding");

    panic_if(!(pkt->isRead() || pkt->isWrite()),
             "Should only see read and writes at memory controller, "
             "saw %s to %#llx\n", pkt->cmdString(), pkt->getAddr());

    // we should not get a new request after committing to retry the
    // current one, but unfortunately the CPU violates this rule, so
    // simply ignore it for now
    if (retryReq)
        return false;

    // if we are busy with a read or write, remember that we have to
    // retry
    if (isBusy) {
        retryReq = true;
        return false;
    }

    // technically the packet only reaches us after the header delay,
    // and since this is a memory controller we also need to
    // deserialise the payload before performing any write operation
    Tick receive_delay = pkt->headerDelay + pkt->payloadDelay;
    pkt->headerDelay = pkt->payloadDelay = 0;
    DPRINTF(SPM, "SPM receive_delay = %llu\n", receive_delay);

    // update the release time according to the bandwidth limit, and
    // do so with respect to the time it takes to finish this request
    // rather than long term as it is the short term data rate that is
    // limited for any real memory

    // calculate an appropriate tick to release to not exceed
    // the bandwidth limit
    Tick duration = pkt->getSize() * bandwidth;
    DPRINTF(SPM, "SPM duration = %llu\n", duration);

    // only consider ourselves busy if there is any need to wait
    // to avoid extra events being scheduled for (infinitely) fast
    // memories
    if (duration != 0) {
        schedule(releaseEvent, curTick() + duration);
        DPRINTF(SPM, "releaseEvent happen in %llu\n", curTick() + duration);
        isBusy = true;
    }

    // go ahead and deal with the packet and put the response in the
    // queue if there is one
    bool needsResponse = pkt->needsResponse();
    recvAtomic(pkt);
    // turn packet around to go back to requestor if response expected
    if (needsResponse) {
        // recvAtomic() should already have turned packet into
        // atomic response
        assert(pkt->isResponse());

        Tick totLat = ((pkt->isRead()) ? getLatency() : 0) + ((pkt->isWrite()) ? getWriteLatency() : 0);
        DPRINTF(SPM, "SPM totLat = %llu\n", totLat);
        Tick when_to_send = curTick() + receive_delay + totLat;
        DPRINTF(SPM, "SPM when_to_send = %llu\n", when_to_send);

        // typically this should be added at the end, so start the
        // insertion sort with the last element, also make sure not to
        // re-order in front of some existing packet with the same
        // address, the latter is important as this memory effectively
        // hands out exclusive copies (shared is not asserted)
        auto i = packetQueue.end();
        --i;
        while (i != packetQueue.begin() && when_to_send < i->tick &&
               !i->pkt->matchAddr(pkt))
            --i;

        // emplace inserts the element before the position pointed to by
        // the iterator, so advance it one step
        packetQueue.emplace(++i, pkt, when_to_send);

        if (!retryResp && !dequeueEvent.scheduled())
            schedule(dequeueEvent, packetQueue.back().tick);
    } else {
        pendingDelete.reset(pkt);
    }

    printf("quit spm recvTimingReq\n");
    return true;
}

Tick
ScratchpadMemory::getWriteLatency() const
{
    
    return latency_write +
        (latency_write_var ? rng_wr->random<Tick>(0, latency_write_var) : 0);
        // (latency_write_var ? random_mt.random<Tick>(0, latency_write_var) : 0);
}

// ScratchpadMemory*
// ScratchpadMemoryParams::create()
// {
//     return new ScratchpadMemory(this);
// }
// ScratchpadMemory* 
// ScratchpadMemoryParams::create() const 
// {
//     return new ScratchpadMemory(*this);
// }

} //namespace memory
} //namespace gem5