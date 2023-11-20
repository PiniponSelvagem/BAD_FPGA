#include <ap_int.h>
#include <ap_axi_sdata.h>
#include "hls_stream.h"

#include "types.h"
#include "size_conv3D.h"

typedef ap_int<W_BIT_WIDTH*PACKET> weigth_t;
typedef ap_int<I_BIT_WIDTH*PACKET> imap_t;
typedef ap_uint<I_BIT_WIDTH*PACKET> omap_t;
typedef ap_int<16> bias_t;
typedef ap_int<13> count_t;
typedef ap_int<40> accum_t;  // I_BIT_WIDTH+W_BIT_WIDTH + 4

typedef ap_axis<64, 0, 0, 0> in_pkt;
typedef ap_axis<64, 0, 0, 0> out_pkt;

// The top-level function
void conv2D(hls::stream<in_pkt> &strm_in, hls::stream<out_pkt> &strm_out, unsigned char scale, int pool) {
#pragma HLS INTERFACE ap_ctrl_none port=return
#pragma HLS interface axis port=strm_in
#pragma HLS INTERFACE axis port=strm_out

    imap_t img_in[(K_SIZE+1)*IWIDTH*IDEPTH/PACKET];
    weigth_t weights[FILTERS*K_SIZE*K_SIZE*IDEPTH/PACKET];
#pragma HLS ARRAY_PARTITION variable=weights type=cyclic factor=2
#pragma HLS ARRAY_PARTITION variable=img_in type=cyclic factor=2

    in_pkt tmp;
    out_pkt tmpo;

    READ_WEIGHTS: for (int i = 0; i < FILTERS*K_SIZE*K_SIZE*IDEPTH/PACKET; i++){    //FILTERS*(K_SIZE*K_SIZE*IDEPTH/PACKET+1
        tmp = strm_in.read();
        weights[i] = tmp.data.range(63, 0);
        printf("weights[%d] = 0x%08x 0x%08x\n", i, (int)weights[i].range(63,32), (int)weights[i].range(31,0));
    }

    READ_INIT_MAP: for (int i = 0; i < (K_SIZE-1)*IWIDTH*IDEPTH/PACKET; i++){
        tmp = strm_in.read();
        img_in[i] = tmp.data.range(63, 0);
        printf("img_in[%d] = 0x%08x 0x%08x\n", i, (int)img_in[i].range(63,32), (int)img_in[i].range(31,0));
    }

    int kk, xx, w_index, i_index, i_index1, rd_index, cp;
    omap_t acc_sat, acc_sat1;
    accum_t acc;
    accum_t accArray[64];

    loop_orow: for(int orow = -1; orow < OHEIGHT-1; orow++){
        //printf("orow = %d\n", (int)orow);
        rd_index = 0;
        loop_ocol: for(int ocol = -1, cp = 1; ocol < OWIDTH-1; ocol += 1, cp += 1){
            //printf("ocol = %d\n", (int)ocol);
    //#pragma HLS PIPELINE
            filters_loop: for(int i = 0; i < FILTERS; i++){
                //printf("i (filter) = %d\n", (int)i);
                accum_t acc = 0;
                w_index = i * K_SIZE * K_SIZE * IDEPTH/PACKET;
                if (rd_index < IWIDTH * IDEPTH/PACKET && orow < OHEIGHT - 3){
                    tmp = strm_in.read();
                    img_in[(orow+3)%4 * IWIDTH*IDEPTH/PACKET + rd_index] = tmp.data.range(63, 0);
                }
                rd_index += 1;
                for (int j = 0; j < K_SIZE; j++){
                    for (int k = 0; k < K_SIZE; k++){
                        for (int l = 0; l < IDEPTH/PACKET; l++){
#pragma HLS UNROLL factor = 4
                            i_index = ((orow+j)%4) * IWIDTH*IDEPTH/PACKET + (ocol+k) * IDEPTH/PACKET + l;
                            //printf("bi = %d\n", (int)i_index);
                            //printf("orow= %d | ocol= %d | i (filter)= %d | j= %d | k= %d | l= %d\n", (int)orow, (int)ocol, (int)i, (int)j, (int)k, (int)l);
                            if (!(orow+j == -1 || ocol+k == -1 || orow+j == OHEIGHT || ocol+k == OWIDTH)){
                                //printf("weights[%d] = 0x%0x 0x%0x\n", w_index, (int)weights[w_index].range(63,32), (int)weights[w_index].range(31,0));
                                //printf("img_in[%d] = 0x%0x 0x%0x\n", i_index, (int)img_in[i_index].range(63,32), (int)img_in[i_index].range(31,0));

                                acc += weights[w_index].range(15,0)  * img_in[i_index].range(15,0);
                                acc += weights[w_index].range(31,16) * img_in[i_index].range(31,16);
                                acc += weights[w_index].range(47,32) * img_in[i_index].range(47,32);
                                acc += weights[w_index].range(63,48) * img_in[i_index].range(63,48);
                                //printf("accin %d, %d, %d, %d\n", (int)i_index, (int)acc, (int)img_in[i_index].range(15,0), (int)img_in[i_index].range(31,16));
                            }
                            w_index += 1;
                        }
                    }
                }

                if (acc <= 0) acc_sat = 0;
                else acc_sat = (acc >> scale);
                //printf("acc %d, %d\n", (int)acc, (int)cp);


                if (pool == 1){
                    if (i%4 == 0)
                        tmpo.data.range(15,0) = acc_sat.range(15,0);
                    if (i%4 == 1)
                        tmpo.data.range(31,16) = acc_sat1.range(15,0);
                    if (i%4 == 2)
                        tmpo.data.range(47,32) = acc_sat.range(15,0);
                    if (i%4 == 3){
                        tmpo.data.range(63,48) = acc_sat1.range(15,0);
                        tmpo.strb = 0xF;
                        tmpo.keep = 0xF;
                        if (orow == OHEIGHT-1 && ocol == OWIDTH-1) tmpo.last = 1;
                        else tmpo.last = 0;
                        strm_out.write(tmpo);
                        //printf("values %d %d\n", (int)tmpo.data.range(15,0), i);
                    }
                    pool = 1;
                }
                else{
                    if (cp == pool){
                        if (acc_sat > accArray[i])
                            accArray[i] = acc_sat;
                        if (i%4 == 0)
                            tmpo.data.range(15,0) = accArray[i].range(15,0);
                        if (i%4 == 1)
                            tmpo.data.range(31,16) = accArray[i].range(15,0);
                        if (i%4 == 2)
                            tmpo.data.range(47,32) = accArray[i].range(15,0);
                        if (i%4 == 3){
                            tmpo.data.range(63,48) = accArray[i].range(15,0);
                            tmpo.strb = 0xF;
                            tmpo.keep = 0xF;
                            if (orow == OHEIGHT-1 && ocol == OWIDTH-1) tmpo.last = 1;
                            else tmpo.last = 0;
                            strm_out.write(tmpo);
                            //printf("values %d %d\n", (int)tmpo.data.range(15,0), i);
                        }
                        if (i == FILTERS-1) cp = 0;
                    }
                    else{
                        if (cp == 1)
                            accArray[i] = acc_sat;
                        else
                            if (acc_sat > accArray[i])
                                accArray[i] = acc_sat;
                    }
                }
            }
        }
    }
}
