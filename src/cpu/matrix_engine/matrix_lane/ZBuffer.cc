//===================================================================================
// Authors: chenwanqi
// Date: 2024/1/18
//===================================================================================
// Description:
// 1. Number of ports is fixed, but active ports will depend on data size.
// 2. Default datatype is int8.
// 3. Default input mode is: output one row per cycle; Zbuffer is just like YBuffer
//===================================================================================
#include "cpu/matrix_engine/matrix_lane/ZBuffer.hh"
#include "debug/ZBuffer.hh"

#include <cassert>
#include <cstdint>

namespace gem5
{
    ZBuffer::ZBuffer(const ZBufferParams &params):
    TickedObject(params, Event::Serialize_Pri), buffer_depth(params.buffer_depth),data_width(params.data_width),
    num_port(params.num_port)
{
    unsigned_writeRequest.resize(num_port);
    signed_writeRequest.resize(num_port);
    unsigned_data_temp.resize(num_port);
    signed_data_temp.resize(num_port);
    unsigned_buffer.resize(num_port);
    signed_buffer.resize(num_port);
    data_ready = false;
    busy = false; 
    full = false;
    sendReq = false;
    send_cnt = 0;
}

ZBuffer::~ZBuffer()
{}

void ZBuffer::set_topPtr(MatrixLane* _matrix_lane, MatrixEngine* _matrix_engine)
{
    matrix_engine = _matrix_engine;
    matrix_lane = _matrix_lane;
}

void ZBuffer::regStats()
{
    TickedObject::regStats();

    read_from_ZBuffer
    .name(name() + ".read_from_ZBuffer")
    .desc("Number of reads from the ZBuffer");
    write_to_ZBuffer
    .name(name() + ".write_to_ZBuffer")
    .desc("Number of writes to the ZBuffer");
    ZBuffer_occupied
    .name(name() + ".ZBuffer_occupied")
    .desc("Number of cycles that ZBuffer is occupied");
}

void ZBuffer::startTicking(uint64_t matrix_column, uint64_t matrix_row, uint8_t idx)
{
    // assert(!busy);
    data_size = matrix_column;
    column_size = matrix_column;
    row_size = matrix_row;
    dstReg_idx = idx;
    DPRINTF(ZBuffer, "ZBuffer is start working!\n");
    start();
    busy = true;

}

uint64_t ZBuffer::get_unsigned_size()
{
    return unsigned_buffer[0].size();
}

uint64_t ZBuffer::get_signed_size()
{
    return signed_buffer[0].size();
}

void ZBuffer::receive_data(uint8_t data, uint32_t coordinate_x, uint32_t coordinate_y)
{
    // assert(!full);
    // if(offset == 0){
    //     unsigned_data_temp[coordinate_y].push_back({0, 0, 0, 0});
    // }
    // std::array<uint8_t, 4>& tail = unsigned_data_temp[coordinate_y].back();
    // tail[offset] = data;
    // if(offset == 3){
    //     write_to_ZBuffer++;
    //     unsigned_writeRequest[coordinate_y] = true;
    // }
    // isSigned = false;
    assert(!full);
    unsigned_data_temp[coordinate_x].push_back(data);
    write_to_ZBuffer++;
    unsigned_writeRequest[coordinate_x] = true;
    isSigned = false;
    data_ready = true;
}

void ZBuffer::receive_data(int8_t data, uint32_t coordinate_x, uint32_t coordinate_y)
{
    // assert(!full);
    // if(offset == 0){
    //     signed_data_temp[coordinate_x].push_back({0, 0, 0, 0});
    // }
    // std::array<int8_t, 4>& tail = signed_data_temp[coordinate_x].back();
    // tail[offset] = data;
    // if(offset == 3){
    //     write_to_ZBuffer++;
    //     signed_writeRequest[coordinate_x] = true;
    //     // signed_data_temp[coordinate_y].push_back(data);
    //     int32_t combined = ((int32_t)tail[0] & 0xFF) | (((int32_t)tail[1] & 0xFF) << 8) | (((int32_t)tail[2] & 0xFF) << 16) | (((int32_t)tail[3] & 0xFF) << 24);

    //     DPRINTF(ZBuffer, "zbuffer receive data = %d\n", combined);
    // }
    // isSigned = true;
    assert(!full);
    signed_data_temp[coordinate_x].push_back(data);
    DPRINTF(ZBuffer, "zbuffer receive data = %d, from y = %d\n", data, coordinate_x);
    write_to_ZBuffer++;
    signed_writeRequest[coordinate_x] = true;
    isSigned = true;
    data_ready = true;
}

/**
 * @description: Actually here, we will always know the output matrix size,is m x n, 
 * @param {uint8_t} dstReg_idx
 * @return {*}
 */
void ZBuffer::send_data(uint8_t dstReg_idx)
{
    if (!matrix_engine || !matrix_engine->matrix_reg) {
        std::cerr << "Error: Uninitialized matrix engine or register." <<       std::endl;
        return;
    }

    for(uint8_t i = 0; i < column_size; i++){
            if(isSigned){
                matrix_engine->matrix_reg->wtreg_byte(dstReg_idx, send_cnt%matrix_engine->matrix_reg->bank_num, send_cnt/matrix_engine->matrix_reg->bank_num, i, signed_buffer[i].front());
            } else {
                matrix_engine->matrix_reg->wtreg_byte(dstReg_idx, send_cnt%matrix_engine->matrix_reg->bank_num, send_cnt/matrix_engine->matrix_reg->bank_num, i, unsigned_buffer[i].front());
            }
        if(isSigned){
            signed_buffer[i].pop_front();
        } else{
            unsigned_buffer[i].pop_front();
        }
        
    }

    send_cnt = send_cnt + 1;
}


void ZBuffer::send_req()
{
    assert(!sendReq);
    sendReq = true;
}

void ZBuffer::stopTicking()
{
    DPRINTF(ZBuffer, "ZBuffer stop working!\n");
    stop();
    busy = false;

}

bool ZBuffer::isBusy()
{
    return busy;
}

bool ZBuffer::isFull()
{
    return ((unsigned_buffer.size() == buffer_depth)||(signed_buffer.size() == buffer_depth));
}

void ZBuffer::evaluate()
{
    // Stage1: process send data first!
    if(sendReq){
        send_data(dstReg_idx);
        data_ready = false;
        if(send_cnt == row_size){
            //send finish
            sendReq = false;
            send_cnt = 0;
            matrix_engine->matrix_reg->rls_wrport();
            // stopTicking();
            busy = false;
        }
    }
    // Stage2: process receive data!
    if(isSigned){
        for (uint32_t i = 0; i < data_size; i++){
            if(signed_writeRequest[i]){
                signed_buffer[i].push_back(signed_data_temp[i].front());
                signed_data_temp[i].pop_front();
            }
        }
    } else {
        for (uint32_t i = 0; i < data_size; i++){
            if(unsigned_writeRequest[i]){
                unsigned_buffer[i].push_back(unsigned_data_temp[i].front());
                unsigned_data_temp[i].pop_front();
            }
        }        
    }

    //Stage3: check full & busy
    full = (unsigned_buffer[0].size() == buffer_depth);
    if(busy){
        ZBuffer_occupied++;
    }
    // Stage3: update data_ready
    // if(isSigned){
    //     data_ready = (signed_buffer[0].size() == row_size);
    // } else {
    //     data_ready = (unsigned_buffer[0].size() == row_size);
    // }
    if(data_ready&&matrix_engine->matrix_reg->occupy_wtport()){
    // if(data_ready){
        send_req();
    }
    //Stage4: reset signal
    if(isSigned){
        for (uint32_t i = 0; i < data_size; i++){
            signed_writeRequest[i] = false;
        }
    } else{
        for (uint32_t i = 0; i < data_size; i++){
            unsigned_writeRequest[i] = false;
        }    
    }
}
} // namespace gem5
