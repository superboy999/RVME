#include "cpu/matrix_engine/common/BaseBuffer.hh"
#include "debug/BaseBuffer.hh"

#include <string>
#include <vector>
#include <deque>
#include <cassert>
#include <cstdint>

namespace gem5
{
BaseBuffer::BaseBuffer(const BaseBufferParams &params) : 
    ClockedObject(params), buffer_depth(params.buffer_depth),
    input_size(params.buffer_depth), output_size(params.output_size),
    data_width(params.data_width), buffer_depth(params.buffer_depth)
{

}

bool BaseBuffer::buffer_full()
{
    return (buffer.size() >= buffer_depth);
}

bool BaseBuffer::buffer_empty()
{
    return (buffer.size() == 0);
}

void BaseBuffer::regStats()
{
    ClockedObject::regStats();
}

void BaseBuffer::print_data()
{
    for (uint64_t i = 0; i < buffer.size(); i++)
    {
        DPRINTF(BaseBuffer, "Data in BaseBuffer: NUMBER:%d, Value: %d\n", i, buffer(i));
    }
}


}
