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
#include <array>
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
    send2EWUReq = false;
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

void ZBuffer::receive_data(uint32_t data, uint32_t coordinate_x, uint32_t coordinate_y)
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
    DPRINTF(ZBuffer, "zbuffer receive data = %d, from x = %d\n", data, coordinate_x);
    write_to_ZBuffer++;
    unsigned_writeRequest[coordinate_x] = true;
    isSigned = false;
    data_ready = true;
}

void ZBuffer::receive_data(int32_t data, uint32_t coordinate_x, uint32_t coordinate_y)
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
    DPRINTF(ZBuffer, "zbuffer receive data = %d, from x = %d\n", data, coordinate_x);
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
    std::array<uint32_t, 8> toEWU = {0};
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

void ZBuffer::send2EWU_req() //这个应该是在zbuffer eval里面调用的
{
    assert(!send2EWUReq);
    send2EWUReq = true;
   if(matrix_engine->ew_unit->isIdle() == true){
        matrix_engine->ew_unit->startTicking();
    }
    matrix_engine->ew_unit->acc_req(dstReg_idx, row_size, matrix_lane->lane_idx, isSigned);
}

std::array<uint32_t, 8> ZBuffer::send2EWU_u() //这个返回值具体操作还是在EWU里面进行吧
{
    std::array<uint32_t, 8> data_out = {0};
    // assert(!read_from_ZBuffer);
    // assert(matrix_engine->ew_unit);
    read_from_ZBuffer++;
    for(uint8_t i = 0; i < column_size; i++){
        assert(!unsigned_buffer[i].empty());
        data_out[i] = unsigned_buffer[i].front();
        unsigned_buffer[i].pop_front();
    }
    send_cnt = send_cnt + 1;
    DPRINTF(ZBuffer, "ZBuffer send to EWU data: %d, %d, %d, %d, %d, %d, %d, %d\n", data_out[0], data_out[1], data_out[2], data_out[3], data_out[4], data_out[5], data_out[6], data_out[7]);
    return data_out;
}

std::array<int32_t, 8> ZBuffer::send2EWU_s() //这个返回值具体操作还是在EWU里面进行吧
{
    std::array<int32_t, 8> data_out = {0};
    // assert(!read_from_ZBuffer);
    // assert(matrix_engine->ew_unit);
    read_from_ZBuffer++;
    for(uint8_t i = 0; i < column_size; i++){
        if(isSigned){
            assert(!signed_buffer[i].empty());
            data_out[i] = signed_buffer[i].front();
            signed_buffer[i].pop_front();
        } 
    }
    send_cnt = send_cnt + 1;
    DPRINTF(ZBuffer, "ZBuffer send to EWU data: %d, %d, %d, %d, %d, %d, %d, %d\n", data_out[0], data_out[1], data_out[2], data_out[3], data_out[4], data_out[5], data_out[6], data_out[7]);
    return data_out;
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
    if(send2EWUReq){
        // send_data(dstReg_idx);
        data_ready = false;
        if(send_cnt == row_size){
            //send finish
            send2EWUReq = false;
            send_cnt = 0;
            // matrix_engine->matrix_reg->rls_wrport(dstReg_idx, 0, send_cnt);
            // matrix_engine->matrix_reg->occupy_rdport(dstReg_idx, 0, send_cnt);
            // stopTicking();
            busy = false;
        }
    }
    // Stage2: process receive data!
    if(isSigned){
        for (uint32_t i = 0; i < data_size; i++){
            if(signed_writeRequest[i]){
                assert(signed_data_temp[i].size() != 0);
                DPRINTF(ZBuffer, "signed_data_temp[%d].front() = %d\n", i, signed_data_temp[i].front());
                if(!signed_buffer[i].empty()){
                    DPRINTF(ZBuffer, "before push signed_buffer[%d].front() = %d\n", i, signed_buffer[i].front());
                }
                DPRINTF(ZBuffer, "signed_data_temp.size = %d, signed_buffer.size = %d\n", signed_data_temp[i].size(), signed_buffer[i].size());
                signed_buffer[i].push_back(signed_data_temp[i].front());
                DPRINTF(ZBuffer, "after push signed_buffer[%d].front() = %d\n", signed_buffer[i].front());
                signed_data_temp[i].pop_front();
            }
        }
    } else {
        for (uint32_t i = 0; i < data_size; i++){
            if(unsigned_writeRequest[i]){
                assert(unsigned_data_temp[i].size() != 0);
                DPRINTF(ZBuffer, "unsigned_data_temp[%d].front() = %u\n", i, unsigned_data_temp[i].front());
                if(!unsigned_buffer[i].empty()){
                    DPRINTF(ZBuffer, "before push unsigned_buffer[%d].front() = %u\n", i, unsigned_buffer[i].front());
                }
                DPRINTF(ZBuffer, "unsigned_data_temp.size = %u, unsigned_buffer.size = %u\n", unsigned_data_temp[i].size(), unsigned_buffer[i].size());
                unsigned_buffer[i].push_back(unsigned_data_temp[i].front());
                DPRINTF(ZBuffer, "after push unsigned_buffer[%d].front() = %u\n", i, unsigned_buffer[i].front());
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
    if(data_ready&&matrix_engine->matrix_reg->occupy_wtport(dstReg_idx, 0, send_cnt)&&matrix_engine->matrix_reg->occupy_rdport(dstReg_idx, 0, send_cnt)){
    // if(data_ready){
        // send_req();
        if(signed_writeRequest[0] || unsigned_writeRequest[0]){
            send2EWU_req();
        }
    }
    //Stage4: reset signals
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
