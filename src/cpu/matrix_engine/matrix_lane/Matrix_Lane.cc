#include "cpu/matrix_engine/matrix_lane/Matrix_Lane.hh"
#include "debug/Matrix_Lane.hh"
// #include "cpu/matrix_engine/common/inst_class.hh"

#include <cassert>
#include <cstdint>
#include <vector>

namespace gem5
{
MatrixLane::MatrixLane(const MatrixLaneParams &params) :
    TickedObject(params), XBuffer(params.XBuffer), YBuffer(params.YBuffer),
    zbuffer(params.zbuffer),  xbuffer_ports_num(params.xbuffer_ports_num),
    ybuffer_ports_num(params.ybuffer_ports_num), cu_row_size(params.cu_row_size), cu_column_size(params.cu_column_size),
    WIDEN(params.WIDEN), compute_units(params.compute_units)
{
    unsigned_xdata_temp.resize(xbuffer_ports_num);
    signed_xdata_temp.resize(xbuffer_ports_num);
    unsigned_ydata_temp.resize(ybuffer_ports_num);
    signed_ydata_temp.resize(ybuffer_ports_num);
    computeDone = false;
    sendtoZ = false;
    start_transZ = false;
    start_output = false;
    // computing = false;
    saveZDone = false;
    transOutDone = false;
    busy = false;
    toZcnt = 0;
    writeXRequest = false;
    writeYRequest = false;
    issue_req = false;
    occupy_x = false;
    occupy_y = false;
    compute_units.resize(params.cu_column_size*params.cu_row_size);
    // const std::vector<ComputeUnit *> &compute_units = params.compute_units;
    XBuffer->set_matrixLanePtr(this, cu_row_size, cu_column_size);
    YBuffer->set_matrixLanePtr(this, cu_row_size, cu_column_size);
    printf("MatrixLane has set, cu_row_size = %d, ptr = %p\n", params.cu_row_size, (void *)this);
}

void MatrixLane::set_matrixEnginePtr(MatrixEngine* _matrix_engine)
{
    matrix_engine = _matrix_engine;
    zbuffer->set_topPtr(this, matrix_engine);
}
// void MatrixLane::issue_inst(MatrixInstClass::InstType inst, MatrixInstClass::DataType data_type, MatrixInstClass::MatrixSize matrix_size, bool signed)
// {
//     assert(!busy);
//     busy = true;
//     lane_minst.inst_type = inst;
//     lane_minst.data_type = data_type;
//     lane_minst.matrix_size = matrix_size;
//     lane_minst.signed = signed;
//     startTicking();
//     //startTicking all that stuff!
// }
MatrixLane::~MatrixLane()
{}
void MatrixLane::issue_inst(ScoreBoard_Entry* matrix_sbe)
{
    // assert(!busy);
    MatrixInstClass* minst = new MatrixInstClass();
    lane_minst = minst;
    lane_minst->inst_type = MatrixInstClass::InstType::MultiplyAndAccumulate; //will be complicated later!
    lane_minst->data_type = MatrixInstClass::DataType::INT8;
    lane_minst->matrix_size.partial_size = matrix_sbe->get_cfg_sizeK();
    lane_minst->matrix_size.num_row = matrix_sbe->get_cfg_sizeM();
    lane_minst->matrix_size.num_column = matrix_sbe->get_cfg_sizeN();
    lane_minst->isSigned = matrix_sbe->_minst->isSigned();
    lane_minst->xreg_idx = matrix_sbe->get_renamed_src1();
    lane_minst->yreg_idx = matrix_sbe->get_renamed_src2();
    lane_minst->zreg_idx = matrix_sbe->get_dst_prf_num();
    lane_minst->load_cnt = 0;
    lane_minst->exe_num = 0;
    lane_minst->instDone = false;
    busy = true;
    startTicking();
    InstNum++;
    issue_req = true;
    //startTicking all that stuff!
}

void MatrixLane::rec_x_matrix(uint8_t PhysRegIdx, uint8_t bank_idx, uint8_t entry_idx)
{
    if(lane_minst->isSigned){
        for(uint32_t i = 0; i < lane_minst->matrix_size.num_row; i++){
            // XBuffer->receive_data(Mreg->outputdata(i), i, lane_minst->MatrixSize.column, lane_minst->MatrixSize.row);
            signed_xdata_temp[i].push(matrix_engine->matrix_reg->rdreg_byte(PhysRegIdx, bank_idx, entry_idx, i));
            writeXRequest = true;
            // DPRINTF(Matrix_Lane, "read x data = %u \n", matrix_engine->matrix_reg->rdreg_byte_int(PhysRegIdx, bank_idx, entry_idx, i));
        }
    } else {
        for(uint32_t i = 0; i < lane_minst->matrix_size.num_row; i++){
            // XBuffer->receive_data(Mreg->outputdata(i), i, lane_minst->MatrixSize.column, lane_minst->MatrixSize.row);
            unsigned_xdata_temp[i].push(matrix_engine->matrix_reg->rdreg_byte(PhysRegIdx, bank_idx, entry_idx, i));
            writeXRequest = true;
        }     
    }
}

void MatrixLane::rec_y_matrix(uint8_t PhysRegIdx, uint8_t bank_idx, uint8_t entry_idx)
{
    if(lane_minst->isSigned){
        for(uint32_t i = 0; i < lane_minst->matrix_size.num_column; i++){
            // XBuffer->receive_data(Mreg->outputdata(i), i, lane_minst->MatrixSize.column, lane_minst->MatrixSize.row);
            signed_ydata_temp[i].push(matrix_engine->matrix_reg->rdreg_byte(PhysRegIdx, bank_idx, entry_idx, i));
            writeYRequest = true;
            // DPRINTF(Matrix_Lane, "read y data = %u \n", matrix_engine->matrix_reg->rdreg_byte_int(PhysRegIdx, bank_idx, entry_idx, i));
        }
    } else {
        for(uint32_t i = 0; i < lane_minst->matrix_size.num_column; i++){
            // XBuffer->receive_data(Mreg->outputdata(i), i, lane_minst->MatrixSize.column, lane_minst->MatrixSize.row);
            unsigned_ydata_temp[i].push(matrix_engine->matrix_reg->rdreg_byte(PhysRegIdx, bank_idx, entry_idx, i));
            writeYRequest = true;
        }     
    }  
}


bool MatrixLane::isOccupied()
{
    return busy;
}

void MatrixLane::startTicking()
{
    // assert(!busy);
    DPRINTF(Matrix_Lane, "Matrix Lane is start working!\n");

    for(uint32_t i = 0; i < cu_row_size*cu_column_size; i++){
        compute_units[i]->startTicking(lane_minst->matrix_size.partial_size);
        compute_units[i]->set_zbufferPtr(this->zbuffer);
    }

    // start X/Y/ZBuffer
    XBuffer->startTicking(lane_minst->matrix_size.num_column, lane_minst->matrix_size.num_row);
    YBuffer->startTicking(lane_minst->matrix_size.num_column, lane_minst->matrix_size.num_row);
    zbuffer->startTicking(lane_minst->matrix_size.num_column, lane_minst->matrix_size.num_row, lane_minst->zreg_idx);
    start();
    busy = true;
    InstDone = false;
}

void MatrixLane::stopTicking()
{
    // assert(busy);
    DPRINTF(Matrix_Lane, "Matrix Lane stop working!\n");

    //change to 1D
    //Start ComputeUnit Array
    // for(uint32_t i = 0; i < cu_row_size; i++){
    //     for(uint32_t j = 0; j < cu_column_size; j++){
    //         compute_units[i][j]->stopTicking();
    //     }
    // }
    for(uint32_t i = 0; i < cu_row_size*cu_column_size; i++){
        compute_units[i]->stopTicking();
    }

    // start X/Y/ZBuffer
    XBuffer->stopTicking();
    YBuffer->stopTicking();
    // ZBuffer->stopTicking();

    stop();
    busy = false;
    InstDone = true;
}

void MatrixLane::evaluate()
{
    // DPRINTF(Matrix_Lane, "Matrix Lane check!\n");
    // assert(busy);
    if(transOutDone){
        //This instruction has finally done here!
        // delete lane_minst;
        // stopTicking();
        tranz_minst.instDone = true;
        start_transZ = false;
        start_output = false;
        saveZDone = false;
        transOutDone = false;
        // transInDone = false;
        toZcnt = 0;
        // toCUcnt = 0;
    } else if(!transOutDone&&saveZDone){
        //Send data from ZBuffer out-of this matrix lane
        //FIXME:  ADD funtion and state machine change logic here!!
        start_output = true;
        if(!zbuffer->isBusy()){
            transOutDone = true;
            start_output = false;
        }
    } else if((!saveZDone&&sendtoZ&&!start_transZ)||(start_transZ)){
        
        // computeDone = false;
        if(!start_transZ){
            sendtoZ = false;
            busy = false;
        }
        start_transZ = true;
        if(toZcnt < tranz_minst.matrix_size.num_row){
            for (uint32_t i = 0; i < tranz_minst.matrix_size.num_column; i++){
                compute_units[toZcnt*cu_row_size + i]->output_data();
            }
            toZcnt = toZcnt + 1;
        } else {
            start_transZ = false;
            saveZDone = true;
        }
    } 
    
    if(!computeDone){
        //check last compute unit status
        if(compute_units[0]->computeDone()){
            computeDone = true;
            // computing = false;
            DPRINTF(Matrix_Lane, "Compute Units compute done!\n");
        }
    } else if (computeDone&&!start_transZ&&!start_output){
            sendtoZ = true;
            tranz_minst = *lane_minst;
            delete lane_minst;
            computeDone = false;
        }
        // if(compute_units[0]->isBusy()){
        //     computing = true;
        // }
        // if(computing){
        //     //除去最后一行和最后一列，别的每一个cu只要在运算，就会向下向右传播数据
        //     for(uint8_t i=0; i < lane_minst->matrix_size.num_row-1; i++){
        //         for(uint8_t j=0; j < lane_minst->matrix_size.num_column-1; j++){
        //             if(compute_units[i*cu_column_size+j]->isBusy()){
        //                 DPRINTF(Matrix_Lane, "CU%d send to CU%d and CU%d\n", i*cu_column_size+j, i*cu_column_size+j+1, (i+1)*cu_column_size+j);
        //                 ComputeUnit::data_pair datapair_temp;
        //                 datapair_temp = compute_units[i*cu_column_size+j]->transfer();
        //                 if(lane_minst->isSigned){
        //                     compute_units[i*cu_column_size+j+1]->get_signed_xdata(datapair_temp.s_xdata);
        //                     compute_units[(i+1)*cu_column_size+j]->get_signed_ydata(datapair_temp.s_ydata);
        //                 } else{
        //                     compute_units[i*cu_column_size+j+1]->get_unsigned_xdata(datapair_temp.u_xdata);
        //                     compute_units[(i+1)*cu_column_size+j]->get_unsigned_ydata(datapair_temp.u_ydata);
        //                 }
        //                 // delete datapair_temp;
        //             }
        //         }
        //     }

        //     //FIX: 对于最后一行和最后一列，仅仅进行单一方向的传输。另外最右下角的那个CU不需要传输任何数据
        //     //在这里分开处理三种情况：最后一行，最后一列，最右下角（其实最右下角不需要处理）
        //     for(uint8_t i = 0; i < lane_minst->matrix_size.num_column-1; i++){
        //         if(compute_units[(lane_minst->matrix_size.num_row-1)*cu_column_size+i]->isBusy()){
        //             DPRINTF(Matrix_Lane, "CU%d send to CU%d\n", (lane_minst->matrix_size.num_row-1)*cu_column_size+i, (lane_minst->matrix_size.num_row-1)*cu_column_size+i+1);
        //             if(lane_minst->isSigned){
        //                 compute_units[(lane_minst->matrix_size.num_row-1)*cu_column_size+i+1]->get_signed_xdata(compute_units[(lane_minst->matrix_size.num_row-1)*cu_column_size+i]->transfer().s_xdata);
        //             } else{
        //                 compute_units[(lane_minst->matrix_size.num_row-1)*cu_column_size+i+1]->get_unsigned_xdata(compute_units[(lane_minst->matrix_size.num_row-1)*cu_column_size+i]->transfer().u_xdata);
        //             }
        //         }
        //     }

        //     for(uint8_t i = 0; i < lane_minst->matrix_size.num_row-1; i++){
        //         if(compute_units[i*cu_column_size+lane_minst->matrix_size.num_column-1]->isBusy()){
        //             DPRINTF(Matrix_Lane, "CU%d send to CU%d\n", i*cu_column_size+lane_minst->matrix_size.num_column-1, (i+1)*cu_column_size+lane_minst->matrix_size.num_column-1);
        //             if(lane_minst->isSigned){
        //                 compute_units[(i+1)*cu_column_size+lane_minst->matrix_size.num_column-1]->get_signed_ydata(compute_units[i*cu_column_size+lane_minst->matrix_size.num_column-1]->transfer().s_ydata);
        //             } else{
        //                 compute_units[(i+1)*cu_column_size+lane_minst->matrix_size.num_column-1]->get_unsigned_ydata(compute_units[i*cu_column_size+lane_minst->matrix_size.num_column-1]->transfer().u_ydata);
        //             }
        //         }
        //     }

        // }

    //process input to X/YBuffer
    if(writeXRequest){
        if(lane_minst->isSigned){
            for(uint32_t i = 0; i < lane_minst->matrix_size.num_row; i++){
                XBuffer->receive_data(signed_xdata_temp[i].front(), i);
                // DPRINTF(Matrix_Lane, " xbuffer real recieve data = %u\n", signed_xdata_temp[i].front(), i);
                signed_xdata_temp[i].pop();
            }
        } else {
            for(uint32_t i = 0; i < lane_minst->matrix_size.num_row; i++){
                XBuffer->receive_data(unsigned_xdata_temp[i].front(), i);
                unsigned_xdata_temp[i].pop();
            }
        }
        writeXRequest = false;
    }

    if(writeYRequest){
        if(lane_minst->isSigned){
            for(uint32_t i = 0; i < lane_minst->matrix_size.num_column; i++){
                YBuffer->receive_data(signed_ydata_temp[i].front(), i);
                // DPRINTF(Matrix_Lane, " ybuffer real recieve data = %u\n", signed_ydata_temp[i].front(), i);
                signed_ydata_temp[i].pop();
            }
        } else {
            for(uint32_t i = 0; i < lane_minst->matrix_size.num_column; i++){
                YBuffer->receive_data(unsigned_ydata_temp[i].front(), i);
                unsigned_ydata_temp[i].pop();
            }
        }
        writeYRequest = false;
    }
    
    if(issue_req){
        if(occupy_x && occupy_y){
            rec_x_matrix(lane_minst->xreg_idx, lane_minst->load_cnt%matrix_engine->matrix_reg->bank_num, lane_minst->load_cnt/matrix_engine->matrix_reg->bank_num);
            rec_y_matrix(lane_minst->yreg_idx, lane_minst->load_cnt%matrix_engine->matrix_reg->bank_num, lane_minst->load_cnt/matrix_engine->matrix_reg->bank_num);
            // DPRINTF(Matrix_Lane, "bank idx = %u, entry idx = %u\n", lane_minst->load_cnt%matrix_engine->matrix_reg->bank_num, lane_minst->load_cnt/matrix_engine->matrix_reg->bank_num);
            lane_minst->load();   
        } else if(!matrix_engine->matrix_reg->check_status(lane_minst->xreg_idx) && !matrix_engine->matrix_reg->check_status(lane_minst->yreg_idx)){
            occupy_x = matrix_engine->matrix_reg->try_occupy(lane_minst->xreg_idx);
            occupy_y = matrix_engine->matrix_reg->try_occupy(lane_minst->yreg_idx);
            rec_x_matrix(lane_minst->xreg_idx, lane_minst->load_cnt%matrix_engine->matrix_reg->bank_num, lane_minst->load_cnt/matrix_engine->matrix_reg->bank_num);
            rec_y_matrix(lane_minst->yreg_idx, lane_minst->load_cnt%matrix_engine->matrix_reg->bank_num, lane_minst->load_cnt/matrix_engine->matrix_reg->bank_num);
            // DPRINTF(Matrix_Lane, "bank idx = %u, entry idx = %u\n", lane_minst->load_cnt%matrix_engine->matrix_reg->bank_num, lane_minst->load_cnt/matrix_engine->matrix_reg->bank_num);
            lane_minst->load();  
        }
        if(lane_minst->load_cnt == lane_minst->matrix_size.partial_size){
            // load finish
            lane_minst->load_done = true;
            matrix_engine->matrix_reg->rls(lane_minst->xreg_idx);
            matrix_engine->matrix_reg->rls(lane_minst->yreg_idx);
            issue_req = false; // reset signal
            occupy_x = false; //reset signal
            occupy_y = false; //reset signal
        }
    }

}

void MatrixLane::regStats(){
    TickedObject::regStats();
    InstNum
        .name(name() + ".InstructionNumber")
        .desc("Count how many instructions this matrix lane has computed!");

}

// void MatrixLane::out_matrix(){

// }

// void MatrixLane::recvInst(ScoreBoard_Entry* matrix_sbe){
//     assert(!busy);
//     matrix_sbe = matrix_sbe;
// }
} // namespace gem5

