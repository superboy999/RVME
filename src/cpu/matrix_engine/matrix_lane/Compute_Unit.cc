#include "cpu/matrix_engine/matrix_lane/Compute_Unit.hh"
#include "debug/Compute_Unit.hh"

#include <cassert>
#include <cstdint>

namespace gem5
{
ComputeUnit::ComputeUnit(const ComputeUnitParams &params) : TickedObject(params, Event::Default_Pri), 
en_activate(params.en_activate), quantization_layer(params.quantization_layer), activation_layer(params.activation_layer)
{
    xdata_ready = false;
    ydata_ready = false;
    isSigned = false;
    WIDEN = 4;
    compute_cnt = 0;
    // transfer_cnt = 0;
    compute_done = false;
    output_cnt = 0;
    output_done = false;
    coordinate[0] = params.coordinatex;
    coordinate[1] = params.coordinatey;
    printf("coordinate = [%d,%d]\n", coordinate[0], coordinate[1]);
    // busy = false;
}

ComputeUnit::~ComputeUnit()
{}

void ComputeUnit::set_zbufferPtr(ZBuffer* _zbuffer)
{
    zbuffer = _zbuffer;
}
// ComputeUnit::ComputeUnitStat::ComputeUnitStat(ComputeUnit* parent) :
//     statistics::Group(parent),
//     ADD_STAT(computeNumber, statistics::units::Count::get(),
//         "Count how many datas this CU will compute!")
// {}

// void ComputeUnit::ComputeUnitStat::regStats()
// {

// }
void ComputeUnit::regStats()
{
    TickedObject::regStats();
    computeNumber
        .name(name() + ".computeNumber")
        .desc("Count how many datas this CU will compute!");
}

void ComputeUnit::get_unsigned_xdata(uint8_t xdata)
{
    u_data_x = xdata;
    isSigned = false;
    xdata_ready = true;
}

void ComputeUnit::get_unsigned_ydata(uint8_t ydata)
{
    u_data_y = ydata;
    isSigned = false;
    ydata_ready = true;
}

void ComputeUnit::get_signed_xdata(int8_t xdata)
{
    DPRINTF(Compute_Unit, "get xdata = %d\n", xdata);
    s_data_x = xdata;
    isSigned = true;
    xdata_ready = true;
}

void ComputeUnit::get_signed_ydata(int8_t ydata)
{
    DPRINTF(Compute_Unit, "get ydata = %d\n", ydata);
    s_data_y = ydata;
    isSigned = true;
    ydata_ready = true;
}

// void ComputeUnit::unsigned_mac(uint8_t data_x, uint8_t data_y)
// {
//     u_data_temp = data_x * data_y;
//     computeNumber = computeNumber + 1;
//     data_ready = true;
//     signed = false;
// }

// void ComputeUnit::signed_mac(int8_t data_x, int8_t data_y)
// {
//     s_data_temp =data_x * data_y;
//     computeNumber = computeNumber + 1;
//     data_ready = true;
//     signed = true;
// }

void ComputeUnit::output_data()
{
    // readRequest = true;

    // ======delete the WIDEN*BYTE transfer data function======
    // uint8_t *unsigned_buf = new uint8_t[WIDEN];
    // int8_t *signed_buf = new int8_t[WIDEN];
    // memcpy(signed_buf, &signed_data, WIDEN);
    // memcpy(unsigned_buf, &unsigned_data, WIDEN);
    // DPRINTF(Compute_Unit, "send signed data = %d\n", signed_data);
    // if(WIDEN == 4){
    //     for(uint8_t i = 0; i < WIDEN; i++){
    //         if(isSigned){
    //             zbuffer->receive_data((int8_t)((int8_t*)signed_buf)[i], coordinate[0], coordinate[1], i); 
    //         } else{
    //             zbuffer->receive_data((uint8_t)((uint8_t*)unsigned_buf)[i], coordinate[0], coordinate[1], i);
    //         }
    //     }
    // }=======================================================
    // if(isSigned){
    //     DPRINTF(Compute_Unit, "[%d,%d]signed data = %d \n", coordinate[0], coordinate[1], signed_data);
    // } else {
    //     DPRINTF(Compute_Unit, "[%d,%d]unsignd data = %u \n", coordinate[0], coordinate[1], unsigned_data);
    // }
    
    // ADD quantize and activation function body 
    if(en_activate){
        if(isSigned){
            activate_signed_data = activation_layer->activate(s_pingpong[output_pp_idx]);
            q_signed_data = quantization_layer->quantize(activate_signed_data);
            if(coordinate[0] > 8){
                assert(coordinate[0] < 8);
                assert(coordinate[1] < 8);
            }
            zbuffer->receive_data(q_signed_data, coordinate[0], coordinate[1]);
        } else {
            activate_unsigned_data = activation_layer->activate(u_pingpong[output_pp_idx]);
            q_unsigned_data = quantization_layer->quantize(activate_unsigned_data);
            zbuffer->receive_data(q_unsigned_data, coordinate[0], coordinate[1]);
        }
    } else {
        if(isSigned){
            q_signed_data = quantization_layer->quantize(s_pingpong[output_pp_idx]);
            zbuffer->receive_data(q_signed_data, coordinate[0], coordinate[1]);
        } else {
            q_unsigned_data = quantization_layer->quantize(u_pingpong[output_pp_idx]);
            zbuffer->receive_data(q_unsigned_data, coordinate[0], coordinate[1]);
        }
    }
    output_cnt = output_cnt + 1;
    output_done = (output_cnt == psum) ? true : false;
    // delete[] unsigned_buf;
    // delete[] signed_buf;
    // readRequest = false;
    if(output_done){
        output_pp_idx = (output_pp_idx + 1) & 1;
        output_cnt = 0;
        output_done = false;
        // stopTicking(); //可以理解为，当所有的部分积都算完并且输出完毕了，这时候就可以stop了！
    }

}

void ComputeUnit::evaluate()
{
    if(computeDone()){
        compute_pp_idx = (compute_pp_idx + 1) & 1;
        compute_cnt = 0;
        signed_data = 0;
        unsigned_data = 0;
    }

    if(xdata_ready&&ydata_ready)
    {
        // busy = true;
        if(isSigned)
        {
            signed_data = s_data_x * s_data_y + signed_data;
            s_pingpong[compute_pp_idx] = signed_data;
            // std::array<int8_t, 2> data = {s_data_x, s_data_y};
            // signed_data_queue.push(data);
            // DPRINTF(Compute_Unit,"[%d, %d]:Signed result = %d\n", coordinate[0], coordinate[1], signed_data);
        } else
        {
            unsigned_data = u_data_x * u_data_y + unsigned_data;
            u_pingpong[compute_pp_idx] = unsigned_data;
            // std::array<uint8_t, 2> data = {u_data_x, u_data_y};
            // unsigned_data_queue.push(data);
            // DPRINTF(Compute_Unit,"[%d, %d]:Unsigned result = %d\n", coordinate[0], coordinate[1], unsigned_data);
        }
        // computeNumber= computeNumber + 2; // FIXME: This imply the mac operation is two micro-operation!
        computeNumber++;
        xdata_ready = false;
        ydata_ready = false;
        compute_cnt = compute_cnt + 1;
    } 

    // if(psum == transfer_cnt)
    // {
    //     busy = false;
    // }
    // else if(readRequest)
    // {
    //     //zbuffer->write_data() FIXME: Add the real quotation function
    //     readRequest = false;
    //     stopTicking();
    // }
}

void ComputeUnit::startTicking(uint32_t partial_num)
{
    DPRINTF(Compute_Unit,"Compute Unit (%d, %d) start working! \n", coordinate[0], coordinate[1]);
    psum = partial_num;
    compute_cnt = 0; //For functional correct, reset this here!
    // transfer_cnt = 0;
    start();
}

void ComputeUnit::stopTicking()
{
    DPRINTF(Compute_Unit,"Compute Unit (%d, %d) stop working! \n", coordinate[0], coordinate[1]);
    stop();
    compute_cnt = 0;
    // transfer_cnt = 0;
    output_cnt = 0;
    output_done = false;
    unsigned_data = 0;
    signed_data = 0;
    q_unsigned_data = 0;
    q_signed_data = 0;
}

bool ComputeUnit::computeDone()
{
    return (psum == compute_cnt);
}

// ComputeUnit::data_pair ComputeUnit::transfer()
// {
//     // uint8_t temp_u_datax, temp_u_datay;
//     // int8_t temp_s_datax, temp_s_datay;
//     std::array<uint8_t, 2> temp_u_data = unsigned_data_queue.front();
//     std::array<int8_t, 2> temp_s_data = signed_data_queue.front();
//     data_pair1.s_xdata = temp_s_data[0];
//     data_pair1.s_ydata = temp_s_data[1];
//     data_pair1.u_xdata = temp_u_data[0];
//     data_pair1.u_ydata = temp_u_data[1];
//     transfer_cnt = transfer_cnt + 1;
//     unsigned_data_queue.pop();
//     signed_data_queue.pop();
//     return data_pair1;
// }

// bool ComputeUnit::isBusy()
// {
//     return busy;
// }
}// namespace gem5

  