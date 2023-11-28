#include <ap_int.h>
#include <ap_axi_sdata.h>
#include "hls_stream.h"

#include "types.h"
#include "size_conv3D.h"

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
    weigth_t weights_scale[CHANNELS/PACKET];
    weigth_t bias[CHANNELS/PACKET];
#pragma HLS ARRAY_PARTITION variable=weights type=cyclic factor=2
#pragma HLS ARRAY_PARTITION variable=img_in type=cyclic factor=2

    in_pkt tmp;
    out_pkt tmpo;

    READ_BIAS: for (int i = 0; i < CHANNELS/PACKET; i++){
        tmp = strm_in.read();
        bias[i] = tmp.data.range(63, 0);
        printf("bias[%d] = 0x%08x 0x%08x\n", i, (int)bias[i].range(63,32), (int)bias[i].range(31,0));
    }
    READ_WEIGHTS_SCALE: for (int i = 0; i < CHANNELS/PACKET; i++){
        tmp = strm_in.read();
        weights_scale[i] = tmp.data.range(63, 0);
        printf("weights_scale[%d] = 0x%08x 0x%08x\n", i, (int)weights_scale[i].range(63,32), (int)weights_scale[i].range(31,0));
    }
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
    int b_index, b0, b1;
    omap_t acc_sat;
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
            	b_index = i/PACKET;
            	b0 = (i % PACKET) * W_BIT_WIDTH;
                b1 = (i % PACKET) * W_BIT_WIDTH + W_BIT_WIDTH-1;
                accum_t acc = bias[b_index].range(b1, b0);
                //printf("b_index %d, acc %d, range(%d,%d)\n", b_index, acc, b1, b0);

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

                                acc += weights[w_index].range(3,0)   * img_in[i_index].range(3,0);
                                acc += weights[w_index].range(7,4)   * img_in[i_index].range(7,4);
                                acc += weights[w_index].range(11,8)  * img_in[i_index].range(11,8);
                                acc += weights[w_index].range(15,12) * img_in[i_index].range(15,12);

                                acc += weights[w_index].range(19,16) * img_in[i_index].range(19,16);
                                acc += weights[w_index].range(23,20) * img_in[i_index].range(23,20);
                                acc += weights[w_index].range(27,24) * img_in[i_index].range(27,24);
                                acc += weights[w_index].range(31,28) * img_in[i_index].range(31,28);

                                acc += weights[w_index].range(35,32) * img_in[i_index].range(35,32);
                                acc += weights[w_index].range(39,36) * img_in[i_index].range(39,36);
                                acc += weights[w_index].range(43,40) * img_in[i_index].range(43,40);
                                acc += weights[w_index].range(47,44) * img_in[i_index].range(47,44);

                                acc += weights[w_index].range(51,48) * img_in[i_index].range(51,48);
                                acc += weights[w_index].range(55,52) * img_in[i_index].range(55,52);
                                acc += weights[w_index].range(59,56) * img_in[i_index].range(59,56);
                                acc += weights[w_index].range(63,60) * img_in[i_index].range(63,60);
                                //printf("accin %d, %d, %d, %d\n", (int)i_index, (int)acc, (int)img_in[i_index].range(15,0), (int)img_in[i_index].range(31,16));
                                //printf("i_index %d, acc %d\n", (int)i_index, (int)acc);
                            }
                            w_index += 1;
                        }
                    }
                }

                /*
                valor
                ----- * 4 = valor inteiro do kernel
                scale
                 */
                /*
                ,3frac   *   ,4frac   =   ,7frac
                 */

                //printf("acc %d, %d\n", (int)acc, (int)cp);
                if (acc <= 0) acc_sat = 0;
                else acc_sat = (acc >> scale);	//load scale here, saber quantos fraccionarios tenho, saber onde esta a virgula
                // verificar se esta a saturar, numeros a esquerda do fraccionario se e maior

                if (pool == 1){
                	//printf("i=%d | i%16=%d\n", i, i%16);
                    if (i%16 == 0)
                    	tmpo.data.range(3,0)   = acc_sat.range(3,0);
                    if (i%16 == 1)
                    	tmpo.data.range(7,4)   = acc_sat.range(3,0);
                    if (i%16 == 2)
                    	tmpo.data.range(11,8)  = acc_sat.range(3,0);
                    if (i%16 == 3)
                    	tmpo.data.range(15,12) = acc_sat.range(3,0);
                    if (i%16 == 4)
                    	tmpo.data.range(19,16) = acc_sat.range(3,0);
                    if (i%16 == 5)
                    	tmpo.data.range(23,20) = acc_sat.range(3,0);
                    if (i%16 == 6)
                    	tmpo.data.range(27,24) = acc_sat.range(3,0);
                    if (i%16 == 7)
                    	tmpo.data.range(31,28) = acc_sat.range(3,0);
                    if (i%16 == 8)
                    	tmpo.data.range(35,32) = acc_sat.range(3,0);
                    if (i%16 == 9)
                    	tmpo.data.range(39,36) = acc_sat.range(3,0);
                    if (i%16 == 10)
                    	tmpo.data.range(43,40) = acc_sat.range(3,0);
                    if (i%16 == 11)
                    	tmpo.data.range(47,44) = acc_sat.range(3,0);
                    if (i%16 == 12)
                    	tmpo.data.range(51,48) = acc_sat.range(3,0);
                    if (i%16 == 13)
                    	tmpo.data.range(55,52) = acc_sat.range(3,0);
                    if (i%16 == 14)
                    	tmpo.data.range(59,56) = acc_sat.range(3,0);
                    if (i%16 == 15){
                        tmpo.data.range(63,60) = acc_sat.range(3,0);
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
                        if (i%16 == 0)
                        	tmpo.data.range(3,0)   = accArray[i].range(3,0);
                        if (i%16 == 1)
                        	tmpo.data.range(7,4)   = accArray[i].range(3,0);
                        if (i%16 == 2)
                        	tmpo.data.range(11,8)  = accArray[i].range(3,0);
                        if (i%16 == 3)
                        	tmpo.data.range(15,12) = accArray[i].range(3,0);
                        if (i%16 == 4)
                        	tmpo.data.range(19,16) = accArray[i].range(3,0);
                        if (i%16 == 5)
                        	tmpo.data.range(23,20) = accArray[i].range(3,0);
                        if (i%16 == 6)
                        	tmpo.data.range(27,24) = accArray[i].range(3,0);
                        if (i%16 == 7)
                        	tmpo.data.range(31,28) = accArray[i].range(3,0);
                        if (i%16 == 8)
                        	tmpo.data.range(35,32) = accArray[i].range(3,0);
                        if (i%16 == 9)
                        	tmpo.data.range(39,36) = accArray[i].range(3,0);
                        if (i%16 == 10)
                        	tmpo.data.range(43,40) = accArray[i].range(3,0);
                        if (i%16 == 11)
                        	tmpo.data.range(47,44) = accArray[i].range(3,0);
                        if (i%16 == 12)
                        	tmpo.data.range(51,48) = accArray[i].range(3,0);
                        if (i%16 == 13)
                        	tmpo.data.range(55,52) = accArray[i].range(3,0);
                        if (i%16 == 14)
                        	tmpo.data.range(59,56) = accArray[i].range(3,0);
                        if (i%16 == 15){
                            tmpo.data.range(63,60) = accArray[i].range(3,0);
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
