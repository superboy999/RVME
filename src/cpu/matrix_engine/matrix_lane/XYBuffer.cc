/*
 * @Author: superboy
 * @Date: 2024-06-20 15:53:52
 * @LastEditTime: 2025-03-16 23:26:45
 * @LastEditors: superboy
 * @Description: model the XBuffer or YBuffer
 * @FilePath: /gem5-rvm/src/cpu/matrix_engine/matrix_lane/XYBuffer.cc
 * 
 */
#include "cpu/matrix_engine/matrix_lane/XYBuffer.hh"
#include "debug/XYBuffer.hh"

#include <cassert>
#include <cstdint>
#include <algorithm>

namespace gem5
{

XYBuffer::XYBuffer(const XYBufferParams &params):
    TickedObject(params, Event::Serialize_Pri), buffer_depth(params.buffer_depth), data_width(params.data_width),
    XY(params.XY), num_port(params.num_port) 
{
    unsigned_writeRequest.resize(num_port);
    signed_writeRequest.resize(num_port);
    unsigned_data_temp.resize(num_port);
    signed_data_temp.resize(num_port);
    unsigned_buffer.resize(num_port);
    signed_buffer.resize(num_port);

    // delay_ctl.resize(num_port);
    // std::fill(delay_ctl.begin(), delay_ctl.end(), false); //reset all bool to false
    // delay_index = 0;

    data_ready = false;
    busy = false; //equals !empty
    full = false;
    // const std::vector<ComputeUnit *> &compute_units = params.compute_units;
    
    // just initialize
    data_size = 0;
    column_size = 0;
    row_size = 0;
    receiving_data = false;
}

XYBuffer::~XYBuffer()
{}

void XYBuffer::set_matrixLanePtr(MatrixLane* _matrix_lane, uint64_t _cu_row_size, uint64_t _cu_column_size)
{
    matrix_lane = _matrix_lane;
    cu_row_size = _cu_row_size;
    cu_column_size = _cu_column_size;
}

void XYBuffer::regStats()
{
    TickedObject::regStats();

    read_from_XYBuffer
    .name(name() + ".read_from_XYBuffer")
    .desc("Number of reads from the XYBuffer");
    write_to_XYBuffer
    .name(name() + ".write_to_XYBuffer")
    .desc("Number of writes to the XYBuffer");
    XYBuffer_occupied
    .name(name() + ".XYBuffer_occupied")
    .desc("Number of cycles that XYBuffer is occupied");
}

void XYBuffer::startTicking(uint64_t matrix_column, uint64_t matrix_row)
{
    // assert(!busy);
    if(XY){
        DPRINTF(XYBuffer, "XBuffer is start working!\n");
    } else{
        DPRINTF(XYBuffer, "YBuffer is start working!\n");
    }
    start();
    busy = true;
    column_size = matrix_column;
    row_size = matrix_row;
    if(XY){
        data_size = row_size;
    } else {
        data_size = column_size;
    }
}

uint64_t XYBuffer::get_unsigned_size()
{
    return unsigned_buffer[0].size();
}

uint64_t XYBuffer::get_signed_size()
{
    return signed_buffer[0].size();
}

void XYBuffer::receive_data(uint8_t data, uint32_t portID)
{
    assert(!full);
    write_to_XYBuffer++;
    unsigned_writeRequest[portID] = true;
    // unsigned_buffer.push(data);
    unsigned_data_temp[portID].push(data);
    isSigned = false;
    receiving_data = true;
    if(XY){
        DPRINTF(XYBuffer, "XBuffer receive unsigned %u, in port: %u\n", data, portID);
    } else{
        DPRINTF(XYBuffer, "YBuffer receive unsigned %u, in port: %u\n", data, portID);
    }
    
}

void XYBuffer::receive_data(int8_t data, uint32_t portID)
{
    assert(!full);
    write_to_XYBuffer++;
    signed_writeRequest[portID] = true;
    // unsigned_buffer.push(data);
    signed_data_temp[portID].push(data);
    isSigned = true;
    receiving_data = true;
    if(XY){
        DPRINTF(XYBuffer, "XBuffer receive signed %d, in port: %u\n", data, portID);
    } else{
        DPRINTF(XYBuffer, "YBuffer receive signed %d, in port: %u\n", data, portID);
    }
}


void XYBuffer::send_data()
{
    //Just send one patch(one column or one row)

    // assert(data_ready);//FIX: change the data transfer type!!
    assert(data_size <= num_port);
    read_from_XYBuffer++;
    if(isSigned){
        // signed datatype
        if(XY){

            for (uint32_t i = 0; i < row_size; i++)
            {
                for (uint32_t j = 0; j < column_size; j++)
                {
                    matrix_lane->compute_units[i*cu_row_size+j]->get_signed_xdata(signed_buffer[i].front());
                    DPRINTF(XYBuffer, "XBuffer send %d to cu:%u\n", signed_buffer[i].front(), i*cu_row_size+j);               
                }
                signed_buffer[i].pop(); 
            }
            // for (uint8_t i = 0; i < row_size; i++)
            // {
            //     if(delay_ctl[i]){
            //         matrix_lane->compute_units[i*cu_row_size]->get_signed_xdata(signed_buffer[i].front());
            //         DPRINTF(XYBuffer, "XBuffer send %d to cu:%u\n", signed_buffer[i].front(), i*cu_row_size);
            //         signed_buffer[i].pop(); 
            //     }
            // }
        } else{
            // YBuffer
            for (uint32_t j = 0; j < column_size; j++)
            {
                for (uint32_t i = 0; i < row_size; i++)
                {
                    matrix_lane->compute_units[j+i*cu_row_size]->get_signed_ydata(signed_buffer[j].front());          
                    DPRINTF(XYBuffer, "YBuffer send %d to cu:%u\n", signed_buffer[j].front(), j+i*cu_row_size);     
                }
                signed_buffer[j].pop();     
            }

            // for (uint8_t i = 0; i < column_size; i++)
            // {
            //     if(delay_ctl[i]){
            //         matrix_lane->compute_units[i]->get_signed_ydata(signed_buffer[i].front());
            //         DPRINTF(XYBuffer, "YBuffer send %d to cu:%u\n", signed_buffer[i].front(), i);
            //         signed_buffer[i].pop();
            //     }
            // }
        }
    } else {
        // unsigned datatype
        if(XY){

            for (uint32_t i = 0; i < row_size; i++)
            {
                for (uint32_t j = 0; j < column_size; j++)
                {
                    matrix_lane->compute_units[i*cu_row_size+j]->get_unsigned_xdata(unsigned_buffer[i].front());      
                    DPRINTF(XYBuffer, "XBuffer send %u to cu:%u\n", unsigned_buffer[i].front(), i*cu_row_size+j);              
                }
                unsigned_buffer[i].pop(); 
            }
            // for (uint8_t i = 0; i < row_size; i++)
            // {
            //     if(delay_ctl[i]){
            //         matrix_lane->compute_units[i*cu_row_size]->get_unsigned_xdata(unsigned_buffer[i].front());
            //         DPRINTF(XYBuffer, "XBuffer send %u to cu:%u\n", unsigned_buffer[i].front(), i*cu_row_size);
            //         unsigned_buffer[i].pop(); 
            //     }
            // }
        } else{

            for (uint32_t j = 0; j < column_size; j++)
            {
                for (uint32_t i = 0; i < row_size; i++)
                {
                    matrix_lane->compute_units[j+i*cu_row_size]->get_unsigned_ydata(unsigned_buffer[j].front());      
                    DPRINTF(XYBuffer, "YBuffer send %u to cu:%u\n", unsigned_buffer[j].front(), j+i*cu_row_size);      
                }
                unsigned_buffer[j].pop();     
            }
            // for (uint8_t i = 0; i < column_size; i++)
            // {
            //     if(delay_ctl[i]){
            //         matrix_lane->compute_units[i]->get_unsigned_ydata(unsigned_buffer[i].front());
            //         DPRINTF(XYBuffer, "YBuffer send %u to cu:%u\n", unsigned_buffer[i].front(), i);
            //         unsigned_buffer[i].pop(); 
            //     }
            // }
        }
    }
}

void XYBuffer::stopTicking()
{
    if(XY){
        DPRINTF(XYBuffer, "XBuffer stop working!\n");
    } else{
        DPRINTF(XYBuffer, "YBuffer stop working!\n");
    }
    stop();
    busy = false;
}

bool XYBuffer::isBusy()
{
    return busy;
}

bool XYBuffer::isFull()
{
    return ((unsigned_buffer[0].size() == buffer_depth)||(signed_buffer[0].size() == buffer_depth));
}

// void XYBuffer::update_delay()
// {
//     if(std::all_of(delay_ctl.begin(), delay_ctl.end(), [](bool v) {return v ==false;} )){
//         delay_ctl[0] = true;
//         delay_index = 1;
//     } else {
//         if (XY){
//             // XBuffer
//             if(delay_index < row_size)
//             {
//                 delay_ctl[delay_index] = true;
//                 delay_index ++;
//             }
//         } else {
//             // YBuffer
//             if(delay_index < column_size)
//             {
//                 delay_ctl[delay_index] = true;
//                 delay_index ++;
//             } else {

//             }
//         }
//     }
// }

// void XYBuffer::check_dataRemain()
// {
//     if(XY){
//         for(uint8_t i = 0; i < row_size; i++){
//            if(isSigned){
//                 if((signed_buffer[i].size() == 0)&&(signed_data_temp[i].size() == 0))
//                     delay_ctl[i] = false;
//            } else {
//                 if((unsigned_buffer[i].size() == 0)&&(unsigned_data_temp[i].size() == 0))
//                     delay_ctl[i] = false;
//            } 
//         }
//     } else {
//         for(uint8_t i = 0; i < column_size; i++){
//            if(isSigned){
//                 if((signed_buffer[i].size() == 0)&&(signed_data_temp[i].size() == 0))
//                     delay_ctl[i] = false;
//            } else {
//                 if((unsigned_buffer[i].size() == 0)&&(unsigned_data_temp[i].size() == 0))
//                     delay_ctl[i] = false;
//            } 
//         }
//     }
// }

// void XYBuffer::reset_delay()
// {
//     std::fill(delay_ctl.begin(), delay_ctl.end(), false);
//     delay_index = 0;
// }

void XYBuffer::evaluate()
{

    // DPRINTF(XYBuffer, "XYBuffer check\n");
    // if(data_ready){
    //     DPRINTF(XYBuffer, "front data ready\n");
    // }
    // if(signed_writeRequest[0]){
    //     DPRINTF(XYBuffer, "front signed_writeRequest ready\n");
    // }

    // Stage1: process send data first!
    // if(data_ready || unsigned_buffer[data_size-1].size() != 0 || signed_buffer[data_size-1].size() != 0){
    if(data_ready){
        // update_delay();
        send_data();
        // if(delay_ctl[data_size-1] == true){
        //     check_dataRemain();
        // }
    }
    // Stage2: process receive data!
    if(isSigned&&receiving_data){
        for (uint32_t i = 0; i < data_size; i++){
            if(signed_writeRequest[i]){
                signed_buffer[i].push(signed_data_temp[i].front());
                signed_data_temp[i].pop();
            }
        }
        //reset signal
        receiving_data = false;
    } else if (receiving_data){
        for (uint32_t i = 0; i < data_size; i++){
            if(unsigned_writeRequest[i]){
                unsigned_buffer[i].push(unsigned_data_temp[i].front());
                unsigned_data_temp[i].pop();
            }
        }        
        //reset signal
        receiving_data = false;
    }

    //Stage3: check full & busy
    full = (unsigned_buffer[0].size() == buffer_depth);
    if(busy){
        XYBuffer_occupied++;
    }
    // Stage3: update data_ready
    bool ready = true;
    if(isSigned){
        for (uint32_t i = 0; i < data_size; i++){
            ready = ready && signed_writeRequest[i];
        }
        data_ready = ready;
    } else {
        for (uint32_t i = 0; i < data_size; i++){
            ready = ready && unsigned_writeRequest[i];
        }
        data_ready = ready;
    }

    // if(data_ready){
    //     DPRINTF(XYBuffer, "data ready\n");
    // }
    // if(signed_writeRequest[0]){
    //     DPRINTF(XYBuffer, "signed_writeRequest ready\n");
    // }

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

} //namespace gem5
