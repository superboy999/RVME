#include "cpu/matrix_engine/transpose_unit/Transpose_Unit.hh"
#include "debug/Transpose_Unit.hh"

#include <cassert>
#include <cstdint>
#include <algorithm>

namespace gem5
{

TransposeUnit::TransposeUnit(const TransposeUnitParams &params):
    TickedObject(params, Event::Serialize_Pri) , buffer_depth(params.buffer_depth),
     num_port(params.num_port) 
{
    // unsigned_writeRequest.resize(num_port);
    // signed_writeRequest.resize(num_port);
    // unsigned_buffer.resize(num_port);
    signed_buffer.resize(num_port);

    // delay_ctl.resize(num_port);
    // std::fill(delay_ctl.begin(), delay_ctl.end(), false); //reset all bool to false
    // delay_index = 0;

    data_ready = false;
    busy = false; //equals !empty
    full = false;
    // const std::vector<ComputeUnit *> &compute_units = params.compute_units;
    
    // just initialize
    column_size = 0;
    row_size = 0;
    receiving_data = false;
}

TransposeUnit::~TransposeUnit()
{}

void TransposeUnit::set_matrixEnginePtr(MatrixEngine* _matrix_engine)
{
    matrix_engine = _matrix_engine;
}

void TransposeUnit::regStats()
{
    TickedObject::regStats();

    read_from_TransposeUnit
    .name(name() + ".read_from_TransposeUnit")
    .desc("Number of reads from the TransposeUnit");
    write_to_TransposeUnit
    .name(name() + ".write_to_TransposeUnit")
    .desc("Number of writes to the TransposeUnit");
    TransposeUnit_occupied
    .name(name() + ".TransposeUnit_occupied")
    .desc("Number of cycles that TransposeUnit is occupied");
}

void TransposeUnit::startTicking(uint64_t matrix_column, uint64_t matrix_row)
{
    // assert(!busy);
    DPRINTF(Transpose_Unit, "Transpose Unit is start working!\n");
    start();
    busy = true;
    column_size = matrix_column;
    send_num = 0;
    recv_num = 0;
    row_size = matrix_row;
}

void TransposeUnit::set_phy_id(uint16_t matrixreg_id)
{
    reg_id = matrixreg_id;
}

// uint64_t TransposeUnit::get_unsigned_size()
// {
//     return unsigned_buffer[0].size();
// }

uint64_t TransposeUnit::get_signed_size()
{
    return signed_buffer[0].size();
}

void TransposeUnit::receive_data(uint32_t portID, int8_t data)
{
    assert(!full);
    write_to_TransposeUnit++;
    // unsigned_buffer.push(data);
    signed_data_temp.push(data);
    isSigned = true;
    receiving_data = true;
    DPRINTF(Transpose_Unit, "Transpose Unit receive signed %d, in port: %u\n", data, portID);
}

void TransposeUnit::send_data()
{
    //Just send one patch(one column or one row)

    read_from_TransposeUnit++;
    for (uint32_t j = 0; j < column_size / row_size; j++)
    {
        for (uint32_t i = 0; i < row_size; i++) 
        {
            this->matrix_engine->matrix_reg->wtreg_byte(reg_id, j, send_num, i, signed_buffer[i].front());
            DPRINTF(Transpose_Unit, "Transpose Unit send %d to reg id : %u\n", signed_buffer[i].front(), j*row_size +i);   
            signed_buffer[i].pop();
        }
    }
}

void TransposeUnit::stopTicking()
{
    DPRINTF(Transpose_Unit, "Transpose Unit stop working!\n");
    stop();
    busy = false;
}

bool TransposeUnit::isBusy()
{
    return busy;
}

bool TransposeUnit::isFull()
{
    return (signed_buffer[0].size() == buffer_depth);
}

void TransposeUnit::evaluate()
{

    // Stage1: process send data first!
    if(data_ready){
        // update_delay();
        send_data();
        send_num++;
        if(send_num == row_size) {
            data_ready = false;
            send_num = 0;
        }
    }
    // Stage2: process receive data!
    if(receiving_data){
        for (uint32_t i = 0; i < row_size - 1; i++){
            while(!signed_buffer[i+1].empty()) {
                signed_buffer[i].push(signed_buffer[i+1].front());
                signed_buffer[i+1].pop();
            }
        }
        while(!signed_data_temp.empty()) {
            signed_buffer[row_size-1].push(signed_data_temp.front());
            signed_data_temp.pop();
        }
        recv_num++;

        //reset signal
        receiving_data = false;
        
    }

    //Stage3: check full & busy
    full = (signed_buffer[0].size() == column_size);
    if(busy){
        TransposeUnit_occupied++;
    }
    // Stage3: update data_ready

    if(recv_num == row_size) {
        data_ready = true;
    }

    // if(data_ready){
    //     DPRINTF(TransposeUnit, "data ready\n");
    // }
    // if(signed_writeRequest[0]){
    //     DPRINTF(TransposeUnit, "signed_writeRequest ready\n");
    // }

}

} //namespace gem5
