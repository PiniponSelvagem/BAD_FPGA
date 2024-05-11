#include <ap_int.h>
#include <ap_fixed.h>
#include <ap_axi_sdata.h>
#include "hls_stream.h"

#include "types.h"
#include "size_conv3D.h"

/* WEIGHTS */
// CONV_0
#include "weights/conv_0_bias.h"
#include "weights/conv_0_kernel.h"
#include "weights/conv_0_kernel_scale.h"
#define CONV_0_POOL     1
#define CONV_0_IN_WIDTH IWIDTH
// CONV_1
#include "weights/conv_1_bias.h"
#include "weights/conv_1_kernel.h"
#include "weights/conv_1_kernel_scale.h"
#define CONV_1_POOL     2
#define CONV_1_IN_WIDTH IWIDTH
// CONV_2
#include "weights/conv_2_bias.h"
#include "weights/conv_2_kernel.h"
#include "weights/conv_2_kernel_scale.h"
#define CONV_2_POOL     1
#define CONV_2_IN_WIDTH IWIDTH_1
// CONV_3
#include "weights/conv_3_bias.h"
#include "weights/conv_3_kernel.h"
#include "weights/conv_3_kernel_scale.h"
#define CONV_3_POOL     2
#define CONV_3_IN_WIDTH IWIDTH_1
// CONV_4
#include "weights/conv_4_bias.h"
#include "weights/conv_4_kernel.h"
#include "weights/conv_4_kernel_scale.h"
#define CONV_4_POOL     1
#define CONV_4_IN_WIDTH IWIDTH_2
// CONV_5
#include "weights/conv_5_bias.h"
#include "weights/conv_5_kernel.h"
#include "weights/conv_5_kernel_scale.h"
#define CONV_5_POOL     10
#define CONV_5_IN_WIDTH IWIDTH_2
/***********/


typedef ap_int<13> count_t;
typedef ap_int<16> accum_t;  // I_BIT_WIDTH+W_BIT_WIDTH + 4
typedef ap_int<4>  scale_t;
typedef ap_uint<6> range_t;	 // range calculations results: 0-63

typedef ap_axis<64, 0, 0, 0> in_pkt;
typedef ap_axis<64, 0, 0, 0> out_pkt;

// The top-level function
void conv2D(hls::stream<in_pkt> &strm_in, hls::stream<out_pkt> &strm_out, int layerIndex) {
#pragma HLS INTERFACE s_axilite port=return bundle=BUS1
//#pragma HLS INTERFACE ap_ctrl_none port=return
#pragma HLS INTERFACE axis port=strm_in
#pragma HLS INTERFACE axis port=strm_out
#pragma HLS INTERFACE s_axilite port=layerIndex bundle=BUS1

	int pool;
	int maxWidth;
    imap_t img_in[(K_SIZE+1)*IWIDTH*CHANNELS/PACKET_CNN];
    //weigth_t weights[FILTERS*K_SIZE*K_SIZE*CHANNELS/PACKET_CNN];
    //weigth_t scales[CHANNELS/PACKET_CNN];
    //accum_t bias[CHANNELS];
//#pragma HLS ARRAY_PARTITION variable=weights type=cyclic factor=4
#pragma HLS ARRAY_PARTITION variable=img_in type=cyclic factor=4

    in_pkt tmp;
    out_pkt tmpo;

    /* Old weights loading
     *
    READ_BIAS: for (int i = 0; i < CHANNELS*BIAS_SIZE/BUS_WIDTH; i++){
        tmp = strm_in.read();
        bias[i*4] = tmp.data.range(15, 0);
        bias[i*4+1] = tmp.data.range(31, 16);
        bias[i*4+2] = tmp.data.range(47, 32);
        bias[i*4+3] = tmp.data.range(63, 48);
        //printf("bias = %d %d %d %d\n", (int)bias[i*4], (int)bias[i*4+1], (int)bias[i*4+2], (int)bias[i*4+3]);
    }

    READ_SCALES: for (int i = 0; i < CHANNELS/PACKET_CNN; i++){
        tmp = strm_in.read();
        scales[i] = tmp.data.range(63, 0);
        //printf("scale[%d] = 0x%08x 0x%08x\n", i, (int)scales[i].range(63,32), (int)scales[i].range(31,0));
    }
    READ_WEIGHTS: for (int i = 0; i < FILTERS*K_SIZE*K_SIZE*CHANNELS/PACKET_CNN; i++){
        tmp = strm_in.read();
        weights[i] = tmp.data.range(63, 0);
        //printf("weights[%d] = 0x%08x 0x%08x\n", i, (int)weights[i].range(63,32), (int)weights[i].range(31,0));
    }
    */

    weigth_t* weights;
    weigth_t* scales;
    accum_t* bias;
    switch (layerIndex) {
    	case 0:
    		pool = CONV_0_POOL;
    		maxWidth = CONV_0_IN_WIDTH;
    		weights = (weigth_t*)kernel_0;
    		scales = (weigth_t*)kernel_0_scale;
    		bias = (accum_t*)bias_0;
    		break;
    	case 1:
    		pool = CONV_1_POOL;
    		maxWidth = CONV_1_IN_WIDTH;
    		weights = (weigth_t*)kernel_1;
    		scales = (weigth_t*)kernel_1_scale;
    		bias = (accum_t*)bias_1;
    		break;
    	case 2:
    		pool = CONV_2_POOL;
    		maxWidth = CONV_2_IN_WIDTH;
    		weights = (weigth_t*)kernel_2;
    		scales = (weigth_t*)kernel_2_scale;
    		bias = (accum_t*)bias_2;
    		break;
    	case 3:
    		pool = CONV_3_POOL;
    		maxWidth = CONV_3_IN_WIDTH;
    		weights = (weigth_t*)kernel_3;
    		scales = (weigth_t*)kernel_3_scale;
    		bias = (accum_t*)bias_3;
    		break;
    	case 4:
    		pool = CONV_4_POOL;
    		maxWidth = CONV_4_IN_WIDTH;
    		weights = (weigth_t*)kernel_4;
    		scales = (weigth_t*)kernel_4_scale;
    		bias = (accum_t*)bias_4;
    		break;
    	case 5:
    	default:
    		pool = CONV_5_POOL;
    		maxWidth = CONV_5_IN_WIDTH;
    		weights = (weigth_t*)kernel_5;
    		scales = (weigth_t*)kernel_5_scale;
    		bias = (accum_t*)bias_5;
    		break;
    }

    /*
    READ_BIAS: for (int i = 0; i < CHANNELS*BIAS_SIZE/BUS_WIDTH; i++){
        printf("bias = %d %d %d %d\n", (int)bias[i*4], (int)bias[i*4+1], (int)bias[i*4+2], (int)bias[i*4+3]);
    }

    READ_SCALES: for (int i = 0; i < CHANNELS/PACKET_CNN; i++){
        printf("scale[%d] = 0x%08x 0x%08x\n", i, (int)scales[i].range(63,32), (int)scales[i].range(31,0));
    }
    READ_WEIGHTS: for (int i = 0; i < FILTERS*K_SIZE*K_SIZE*CHANNELS/PACKET_CNN; i++){
        printf("weights[%d] = 0x%08x 0x%08x\n", i, (int)weights[i].range(63,32), (int)weights[i].range(31,0));
    }
    */

    READ_INIT_MAP: for (int i = 0; i < (K_SIZE-1)*maxWidth*CHANNELS/PACKET_CNN; i++){
        tmp = strm_in.read();
        img_in[i] = tmp.data.range(63, 0);
        //printf("img_in[%d] = 0x%08x 0x%08x\n", i, (int)img_in[i].range(63,32), (int)img_in[i].range(31,0));
    }



    int kk, xx, w_index, i_index, i_index1, rd_index, cp;
    int b_index, b0, b1;
    ap_ufixed<4,0> accF;
    accum_t acc_sat;
    accum_t acc, acc_aux, acc_aux1;
    accum_t accArray[64];
    bool even;
    int ax, ocol, orow;

    loop_orow: for(orow = -1; orow < OHEIGHT-1; orow++){
#pragma HLS LOOP_TRIPCOUNT max=431
    	rd_index = 0;
        loop_ocol: for(ocol = -1, cp = 1; ocol < maxWidth-1; ocol += 1, cp += 1){   // TODO: changed
        //loop_ocol: for(ocol = -1, cp = 1; ocol < OWIDTH-1; ocol += 1, cp += 1){
#pragma HLS LOOP_TRIPCOUNT max=40
    //#pragma HLS PIPELINE
            filters_loop: for(int i = 0; i < FILTERS; i++){
                accum_t acc = bias[i];

                w_index = i * K_SIZE * K_SIZE * CHANNELS/PACKET_CNN;
                if (rd_index < maxWidth * CHANNELS/PACKET_CNN && orow < OHEIGHT - 3){
                    tmp = strm_in.read();
                    img_in[(orow+3)%4 * maxWidth*CHANNELS/PACKET_CNN + rd_index] = tmp.data.range(63, 0);
                }
                rd_index += 1;
                loop_kernel_j: for (int j = 0; j < K_SIZE; j++){
//#pragma HLS UNROLL factor = 1
                	loop_kernel_k: for (int k = 0; k < K_SIZE; k++){
#pragma HLS UNROLL factor = 1
                    	loop_calc: for (int l = 0; l < CHANNELS/PACKET_CNN; l++){
//#pragma HLS UNROLL factor = 4
                            i_index = ((orow+j)%4) * maxWidth*CHANNELS/PACKET_CNN + (ocol+k) * CHANNELS/PACKET_CNN + l;
                            //printf("bi = %d\n", (int)i_index);
                            //printf("orow= %d | ocol= %d | i (filter)= %d | j= %d | k= %d | l= %d\n", (int)orow, (int)ocol, (int)i, (int)j, (int)k, (int)l);
                            if (!(orow+j == -1 || ocol+k == -1 || orow+j == OHEIGHT || ocol+k == maxWidth)){
                                //printf("weights[%d] = 0x%0x 0x%0x\n", w_index, (int)weights[w_index].range(63,32), (int)weights[w_index].range(31,0));
                                //printf("img_in[%d] = 0x%0x 0x%0x\n", i_index, (int)img_in[i_index].range(63,32), (int)img_in[i_index].range(31,0));

                            	/* USED IN DEBUG PRINTS AT THE END OF THIS IF SECTION */
                            	//ap_int<4> debug_W;
                            	//ap_fixed<4,1> debug_Wf;
								//ap_int<4> debug_I;
                            	//ap_fixed<4,1> debug_If;
                                //debug_W.range(3,0) = (ap_int<4>)weights[w_index].range(3,0);
                                //debug_Wf.range(3,0) = (ap_int<4>)weights[w_index].range(3,0);
                                //debug_I.range(3,0) = img_in[i_index].range(3,0);
                                //debug_If.range(3,0) = img_in[i_index].range(3,0);
                                /******************************************************/

                                acc += (ap_int<4>)weights[w_index].range(3,0)   * img_in[i_index].range(3,0);
                                acc += (ap_int<4>)weights[w_index].range(7,4)   * img_in[i_index].range(7,4);
                                acc += (ap_int<4>)weights[w_index].range(11,8)  * img_in[i_index].range(11,8);
                                acc += (ap_int<4>)weights[w_index].range(15,12) * img_in[i_index].range(15,12);

                                acc += (ap_int<4>)weights[w_index].range(19,16) * img_in[i_index].range(19,16);
                                acc += (ap_int<4>)weights[w_index].range(23,20) * img_in[i_index].range(23,20);
                                acc += (ap_int<4>)weights[w_index].range(27,24) * img_in[i_index].range(27,24);
                                acc += (ap_int<4>)weights[w_index].range(31,28) * img_in[i_index].range(31,28);

                                acc += (ap_int<4>)weights[w_index].range(35,32) * img_in[i_index].range(35,32);
                                acc += (ap_int<4>)weights[w_index].range(39,36) * img_in[i_index].range(39,36);
                                acc += (ap_int<4>)weights[w_index].range(43,40) * img_in[i_index].range(43,40);
                                acc += (ap_int<4>)weights[w_index].range(47,44) * img_in[i_index].range(47,44);

                                acc += (ap_int<4>)weights[w_index].range(51,48) * img_in[i_index].range(51,48);
                                acc += (ap_int<4>)weights[w_index].range(55,52) * img_in[i_index].range(55,52);
                                acc += (ap_int<4>)weights[w_index].range(59,56) * img_in[i_index].range(59,56);
                                acc += (ap_int<4>)weights[w_index].range(63,60) * img_in[i_index].range(63,60);
                                //printf("accin %d, %d, %d, %d\n", (int)i_index, (int)acc, (int)img_in[i_index].range(15,0), (int)img_in[i_index].range(31,16));
                                //printf("i_index %d, acc %d\n", (int)i_index, (int)acc);
                                //printf("i %d | i_index %d, acc %d, w %f\n", i, (int)i_index, (int)acc, (float)debug_W);
                                //printf("i %d | i_index %d, acc %d, w %f\n", i, (int)i_index, (int)acc, (float)((ap_fixed<4,1>)weights[w_index].range(3,0)));

                                //printf("i %d | i_index %d, acc %d = i %d (%f) * w %d (%f)\n", i, (int)i_index, (int)acc, (int)debug_I, (float)debug_If, (int)debug_W, (float)debug_Wf);
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

                //printf("acc %d - i %d\n", (int)acc, i);

                range_t scaleStart = (i % PACKET_CNN) * 4;
                range_t scaleEnd   = scaleStart + 3;
                scale_t scale = scales[i/PACKET_CNN].range(scaleEnd, scaleStart);
                //printf("%d - %d,%d\n", i, scaleStart, scaleEnd);

                /*
                 * More complex rounding to nearest even, closer to QKeras
                 */
                if (acc <= 0) acc_sat = 0;
                else {
                	acc_aux = (acc >> (scale-1));		//TODO: -1 a todos os scales no python (por causa do bit anterior etc)
                	acc_aux1 = acc_aux;
                	acc_aux = (acc_aux << (scale-1));
                	acc_aux = acc - acc_aux;
                	if (acc_aux1[0] == 1){
                		if (acc_aux != 0 or acc_aux1[1] == 1)
                			acc_sat = acc_aux1 + 2;
                		else
                			acc_sat = acc_aux1;
                	}
                	else
                		acc_sat = acc_aux1;
                }


                /*
                if (acc <= 0) acc_sat = 0;
                else acc_sat = (acc >> (scale-1));		//TODO: -1 a todos os scales no python (por causa do bit anterior etc)

                if (acc_sat[0] == 1){
                	acc_sat = acc_sat + 2;
                }
                */

                acc_sat = acc_sat.range(8,1);
                if (acc_sat > 15)
                	acc_sat = 15;

                accF.range(3,0) = acc_sat.range(3,0);
				//printf("j %d, i %d | acc_sat %2d | accF %f\n", orow, i, (int)acc_sat, (float)accF);


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
                        tmpo.strb = 0xFF;
                        tmpo.keep = 0xFF;
                        if (orow == OHEIGHT-2 && ocol == maxWidth-2 && i == FILTERS-1) tmpo.last = 1;   // TODO: changed
                        //if (orow == OHEIGHT-1 && ocol == maxWidth-1) tmpo.last = 1;
                        else tmpo.last = 0;
                        strm_out.write(tmpo);

                        //accF.range(3,0) = acc_sat.range(3,0);
                        //printf("values %d %d %f\n", i, (int)acc_sat.range(3,0), (float)accF);
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
                            tmpo.strb = 0xFF;
                            tmpo.keep = 0xFF;
                            if (orow == OHEIGHT-2 && ocol == maxWidth-2 && i == FILTERS-1) tmpo.last = 1;   // TODO: changed
                            //if (orow == OHEIGHT-1 && ocol == maxWidth-1) tmpo.last = 1;
                            else tmpo.last = 0;
                            strm_out.write(tmpo);

                            //accF.range(3,0) = accArray[i].range(3,0);
                            //printf("values %d %d %f\n", i, (int)accArray[i].range(3,0), (float)accF);
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

            // TODO: changed
            //if (ocol == maxWidth-2)	//-2 because it can only do a 3x3 until 38, 18, etc
            //	ocol = OWIDTH;
        }
    }

}
