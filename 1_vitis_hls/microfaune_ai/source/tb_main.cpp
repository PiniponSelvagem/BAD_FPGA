#define HW_IP 1
#include "hls_stream.h"
#include <ap_axi_sdata.h>
#include <ap_int.h>

#include "types.h"
#include "size_conv3D.h"

#include "load_weights.h"

typedef ap_axis<64, 0, 0, 0> in_pkt;
typedef ap_axis<64, 0, 0, 0> out_pkt;

hls::stream<in_pkt> str_in;
hls::stream<out_pkt> str_out;

/*
static int bias = 5;
static imap_t image[IHEIGHT][IWIDTH];
static weigth_t kernel[FILTERS][K_SIZE][K_SIZE][IDEPTH/PACKET];
*/

int pf = 1; //10;

void conv2D(hls::stream<in_pkt> &strm_in, hls::stream<out_pkt> &strm_out, unsigned char scale, int pool);


weigth_t input_2[FILTERS*IHEIGHT*IWIDTH/PACKET];

weigth_t kernel_1[FILTERS*CHANNELS*K_SIZE*K_SIZE/PACKET];
weigth_t kernel_1_scale[CHANNELS/PACKET];
weigth_t bias_1[CHANNELS/PACKET];


int main() {
    loadWeights(input_2, kernel_1, kernel_1_scale, bias_1);
    /*
    printf("\ninput_2:\n");
    for (int idx=0; idx<FILTERS*IHEIGHT*IWIDTH/PACKET; idx++) {
    	printf("idx=%d | 0x%016llx\n", idx, input_2[idx]);
    }
    printf("\nkernel_1_scale:\n");
    for (int idx=0; idx<CHANNELS/PACKET; idx++) {
    	printf("idx=%d | 0x%016llx\n", idx, kernel_1_scale[idx]);
    }
    printf("\nbias:\n");
    for (int idx=0; idx<CHANNELS/PACKET; idx++) {
    	printf("idx=%d | 0x%016llx\n", idx, bias_1[idx]);
    }
    printf("\n");
	*/

    int i, j, err_cnt = 0;

    in_pkt tmp;
    out_pkt tmpo;

    for (i=0; i<(CHANNELS/PACKET); i++) {
        tmp.data = bias_1[i];
    	//tmp.data = (weigth_t)0x1111111111111111;
        str_in.write(tmp);
    }
    for (i=0; i<(CHANNELS/PACKET); i++) {
        tmp.data = kernel_1_scale[i];
        str_in.write(tmp);
    }
    for (i=0; i<(FILTERS*K_SIZE*K_SIZE*IDEPTH/PACKET); i++) {
        //printf("%d\n", i);
    	/*
    	if ((i+5) % 9 == 0)
            tmp.data = (weigth_t)0x1111111111111111;
        else
            tmp.data = (weigth_t)0x0000000000000000;
    	*/
        tmp.data = kernel_1[i];
        if (i==(FILTERS*K_SIZE*K_SIZE*IDEPTH/PACKET-1)) tmp.last = (ap_int<1>)1;
        else tmp.last = (ap_int<1>)0;
        str_in.write(tmp);
    }

    for (i=0; i<IHEIGHT*IWIDTH*IDEPTH/PACKET; i++) {
    	//printf("input[%d]\n", i);
    	//tmp.data = (imap_t)0x1110000000000000;
    	tmp.data = input_2[i];
    	/*
    	if (i==0) { // || i==1 || i==2 || i==3) {
            //tmp.data = (imap_t)0x0004000300020001;
            tmp.data = (imap_t)0x0FEDCBA987654321;
        }
        else if (i==5) {
            tmp.data = (imap_t)0x123456789ABCDEF0;
        }
        else {
            tmp.data = (imap_t)0x0101010101010101;
        }
        */
        /*
        if (i/(IDEPTH/PACKET)%2 == 0)
            tmp.data = (imap_t)0x0002000200020002;
        else
            tmp.data = (imap_t)0x0001000100010001;
        */
        if (i == (IHEIGHT*IWIDTH*IDEPTH/PACKET-1)) tmp.last = (ap_int<1>)1;
        else tmp.last = (ap_int<1>)0;
        str_in.write(tmp);
        //printf("%d\n", (int)tmp.data.range(15,0));
    }

    printf("--- END OF SEND ---\n\n");

#if HW_IP
    conv2D(str_in, str_out, 3, pf);
#endif

    for (i=0; i<OWIDTH*OHEIGHT*FILTERS/PACKET/pf; i++) {
        tmpo = str_out.read();
        /*
        printf("%02d - result %d, %d, %d, %d\n", i, (int)tmpo.data.range(15,0), (int)tmpo.data.range(31,16),
                (int)tmpo.data.range(47,32), (int)tmpo.data.range(63,48));
        */
        printf("%02d - result %d, %d, %d, %d,   %d, %d, %d, %d,   %d, %d, %d, %d,   %d, %d, %d, %d,\n", i,
        		(int)tmpo.data.range(3,0),   (int)tmpo.data.range(7,4),   (int)tmpo.data.range(11,8),  (int)tmpo.data.range(15,12),
				(int)tmpo.data.range(19,16), (int)tmpo.data.range(23,20), (int)tmpo.data.range(27,24), (int)tmpo.data.range(31,28),
				(int)tmpo.data.range(35,32), (int)tmpo.data.range(39,36), (int)tmpo.data.range(43,40), (int)tmpo.data.range(47,44),
				(int)tmpo.data.range(51,48), (int)tmpo.data.range(55,52), (int)tmpo.data.range(59,56), (int)tmpo.data.range(63,60)
		);
    }


    return 0;
}
