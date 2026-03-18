#ifndef __CPU_BASE_BUFFER_HH__
#define __CPU_BASE_BUFFER_HH__

#include <string>
#include <vector>
#include <deque>
#include <bitset>
#include <cassert>
#include <cstdint>


#include "sim/clocked_object.hh"
#include "params/BaseBuffer.hh"
// #include "cpu/matrix_engine/matrix_engine.hh"
#include "base/statistics.hh"

// class MatrixEngine;
namespace gem5
{

class BaseBuffer : public ClockedObject
{
public:
    BaseBuffer(const BaseBufferParams &params);
    ~BaseBuffer();

    // virtual void startTicking(MatrixEngine &matrix_wrapper);
    // virtual void startTicking();
    // virtual void stopTicking();

    virtual bool buffer_full();
    virtual bool buffer_empty();

    // ClockedObject already has the function
    virtual void regStats() override;
    // virtual void evaluate() override;
    
    virtual void print_data();

private:
    // MatrixEngine* matrix_wrapper; 之所以要引用这个，是因为要用这个engine里面的函数，先创建一个指向该类型的指针，之后再在函数输入变量声明的时候使用&引用一下，然后用指针指向那个引用就可以了
    // for python configuration
    const uint64_t input_size = 512;
    const uint64_t output_size = 512;
    const uint64_t data_width = 512; // record the width of the data stored in the deque
    const uint64_t buffer_depth = 4;

    struct Bits_def {
        std::bitset<data_width> bits;
    }

    std::deque<Bits_def> buffer;
public:
    // Stats::Scalar 
};
}

#endif