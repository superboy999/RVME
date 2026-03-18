#include "cpu/matrix_engine/matrix_lane/Adder_Tree.hh"
#include "debug/Adder_Tree.hh"

#include <cassert>
#include <cstdint>

namespace gem5
{
AdderTree::AdderTree(const AdderTreeParams &params) : TickedObject(params, Event::Default_Pri), 
quantization_layer(params.quantization_layer), activation_layer(params.activation_layer), en_activate(params.en_activate)
{
    y_data_cnt = 0;
    ydata_ready = false;
    recv_ydata_cnt = 0;
    xdata_ready.resize(params.adder_tree_column_size);
    isSigned = false;
    WIDEN = 4;
    compute_cnt = 0;
    computenum = 0;
    // transfer_cnt = 0;
    compute_done = false;
    output_cnt = 0;
    signed_data.resize(params.adder_tree_column_size)
    for(int i = 0; i < params.adder_tree_column_size; i++) {
        signed_data[i].resize((uint32_t)log2(params.adder_tree_row_size));
    }
    output_done = false;
    adder_tree_row_size = params.adder_tree_row_size;
    adder_tree_column_size = params.adder_tree_column_size;
    s_data_x.resize(params.adder_tree_column_size);
    s_data_y.resize(params.adder_tree_column_size);
    s_data_temp_x.resize(params.adder_tree_row_size);
    s_data_temp_y.resize(params.adder_tree_column_size);
    for (int i = 0; i < params.adder_tree_column_size; i++) {
        s_data_x[i].resize(params.adder_tree_row_size);
        s_data_y[i].resize(params.adder_tree_row_size);
    }
    // busy = false;
}

AdderTree::~AdderTree()
{}


void AdderTree::set_cpu_ptr(gem5::o3::CPU* _o3cpu)
{
    o3cpu = _o3cpu;
}

void AdderTree::set_zfpbufferPtr(ZFPBuffer* _zfpbuffer)
{
    zfpbuffer = _zfpbuffer;
}
// AdderTree::Adder_TreeStat::Adder_TreeStat(Adder_Tree* parent) :
//     statistics::Group(parent),
//     ADD_STAT(computeNumber, statistics::units::Count::get(),
//         "Count how many datas this CU will compute!")
// {}

// void AdderTree::Adder_TreeStat::regStats()
// {

// }
void AdderTree::regStats()
{
    TickedObject::regStats();
    computeNumber
        .name(name() + ".computeNumber")
        .desc("Count how many datas this CU will compute!");
}


bool AdderTree::signed_ydata_ready() {
    return ydata_ready;
}

void AdderTree::get_signed_xdata(int8_t xdata, uint32_t row_num)
{
    DPRINTF(Adder_Tree, "AdderTree : get xdata = %d\n", xdata);
    assert(row_num < adder_tree_row_size);
    s_data_temp_x[row_num] = xdata;
    isSigned = true;
    recv_xdata = true;
}

bool AdderTree::outputdata_ready() {
    return !signed_data[0][0].empty();
}

void AdderTree::get_signed_ydata(int8_t ydata, uint32_t column_num)
{
    DPRINTF(Adder_Tree, "get ydata = %d\n", ydata);
    assert(column_num < adder_tree_column_size);
    s_data_temp_y[column_num] = ydata;
    isSigned = true;
    recv_ydata = true;
}

void AdderTree::output_data()
{
    for(int i = 0; i < column_size; i++) {
        zfpbuffer->receive_data((*reinterpret_cast<uint32_t*>(&signed_data[i][0].front())), i, output_cnt);
        signed_data[i][0].pop();
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
    
    compute_cnt = compute_cnt + 1;
}

float AdderTree::fp_mul(uint8_t x, uint8_t y) {
    uint32_t x_sign = x & (128);
    uint32_t x_exp = ((x >> 2) & 31) + 112;
    uint32_t x_mant = (x & 3) << 21;
    if(((x >> 2) & 31) == 0) {
        if(x_mant == 0) {
            x_exp = 0;
        }
        else {
            while(x_mant < (1 << 23)) {
                x_exp --;
                x_mant = x_mant << 1;
            }
            x_exp = x_exp + 1;
            x_mant = x_mant - (1 << 23);
        }
    }
    else if(((x >> 2) & 31) == 31) {
        x_exp = 255;
    }
    uint32_t x_data = (x_sign << 31) + (x_exp << 23) + x_mant;

    uint32_t y_sign = y & (128);
    uint32_t y_exp = ((y >> 2) & 31) + 112;
    uint32_t y_mant = (y & 3) << 21;
    if(((y >> 2) & 31) == 0) {
        if(y_mant == 0) {
            y_exp = 0;
        }
        else {
            while(y_mant < (1 << 23)) {
                y_exp --;
                y_mant = y_mant << 1;
            }
            y_exp = y_exp + 1;
            y_mant = y_mant - (1 << 23);
        }
    }
    else if(((y >> 2) & 31) == 31) {
        y_exp = 255;
    }
    uint32_t y_data = (y_sign << 31) + (y_exp << 23) + y_mant;
    return (*reinterpret_cast<float*>(&x_data)) * (*reinterpret_cast<float*>(&y_data));
}

void AdderTree::evaluate()
{
    if(computeDone()){
        compute_pp_idx = (compute_pp_idx + 1) & 1;
        compute_cnt = 0;
        for(int j = 0; j < adder_tree_column_size; j++) {
            for(int i = 0; i < (uint32_t)log2(adder_tree_row_size); i++) {
                while(!signed_data[j][i].empty())
                    signed_data[j][i].pop();
            }
        }
    }

    for(int j = 0; j < adder_tree_column_size; j++) {
        for(int i = 1; i < (uint32_t)log2(adder_tree_row_size); i++) {
            if(signed_data[j][i].size()) {
                assert(signed_data[j][i].size() % 2 == 0);
                for(int idx = 0; idx < pow(2, i); idx = idx + 2) {
                    float float_add = signed_data[j][i].front(); signed_data[j][i].pop();
                    float_add = float_add + signed_data[j][i].front(); signed_data[j][i].pop();
                    signed_data[j][i-1].push(float_add);
                }
            }
        }
    }

    if(recv_ydata) {
        for(int j = 0; j < adder_tree_column_size; j++) {
            for(int i = adder_tree_row_size - 1; i > 0; i--) {
                s_data_y[j][i] = s_data_y[j][i-1];
            }
            s_data_y[j][0] = s_data_temp_y[j];
        }
        recv_ydata_cnt++;
        if(recv_ydata_cnt == adder_tree_row_size) {
            ydata_ready = true;
        }
        else {
            ydata_ready = false;
        }
    }
    
    for (int j = adder_tree_column_size - 1; j > 0; j--) {
        xdata_ready[j] = xdata_ready[j-1];   
    }
    xdata_ready[0] = recv_xdata;

    for(int j = adder_tree_column_size - 1; j > 0; j--) {
        if(xdata_ready[j]) {
            if(j > 0) {
                for(int i = 0; i < adder_tree_row_size; i++) {
                    s_data_x[j][i] = s_data_x[j-1][i];
                }
            }
            else {
                for(int i = 0; i < adder_tree_row_size; i++) {
                    s_data_x[j][i] = s_data_temp_x[i];
                }
            }
        }
    }
    recv_xdata = false; recv_ydata = false;

    s_pingpong[compute_pp_idx] = signed_data[0].front();

    for(int j = 0; j < adder_tree_column_size; j++) {
        if(xdata_ready[j] && ydata_ready)
        {
            for(int i = 0; i < adder_tree_row_size; i += 2) {
                float floatpsum = fp_mul(s_data_x[j][i], s_data_y[j][i]) + fp_mul(s_data_x[j][i+1], s_data_y[j][i+1]);
                signed_data[j][(uint32_t)log2(adder_tree_row_size) - 1].push(floatpsum);
            }
            computeNumber++; 
            computenum++;
            xdata_ready[j] = false;
            // ydata_ready = false;
        }
    }
    

}

void AdderTree::startTicking(uint32_t partial_num)
{
    DPRINTF(Adder_Tree,"Adder Tree start working! \n");
    psum = partial_num;
    // transfer_cnt = 0;
    start();
}

void AdderTree::stopTicking()
{
    DPRINTF(Adder_Tree,"Adder Tree stop working! \n");
    stop();
    compute_cnt = 0;
    // transfer_cnt = 0;
    output_cnt = 0;
    output_done = false;
    // unsigned_data = 0;
    signed_data.clear();
    // q_unsigned_data = 0;
    // q_signed_data = 0;
    y_data_cnt = 0;
}

bool AdderTree::computeDone()
{
    return (psum == compute_cnt);
}

bool AdderTree::isBusy()
{
    return busy;
}

}// namespace gem5

  