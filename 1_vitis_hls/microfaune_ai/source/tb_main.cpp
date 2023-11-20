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


weigth_t kernel_1[FILTERS*CHANNELS*K_SIZE*K_SIZE/PACKET];


int main() {
    loadWeights(kernel_1);

    int i, j, err_cnt = 0;

    in_pkt tmp;
    out_pkt tmpo;

    for (i=0; i<(FILTERS*K_SIZE*K_SIZE*IDEPTH/PACKET); i++) {
        //printf("%d\n", i);
        /*
        if (i == 4)
            tmp.data = (weigth_t)0x0001000100010001;
        else
            tmp.data = (weigth_t)0x0000000000000000;
        */
        tmp.data = kernel_1[i];
        //tmp.data = (weigth_t)0x0001000100010001;
        if (i==(FILTERS*K_SIZE*K_SIZE*IDEPTH/PACKET-1)) tmp.last = (ap_int<1>)1;
        else tmp.last = (ap_int<1>)0;
        str_in.write(tmp);
    }

    for (i=0; i<IHEIGHT*IWIDTH*IDEPTH/PACKET; i++) {
        if (i==0) { // || i==1 || i==2 || i==3) {
            tmp.data = (imap_t)0x0004000300020001;
            //tmp.data = (imap_t)0x0004000300020001;
        }
        else {
            tmp.data = (imap_t)0x0005000500050005;
        }
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

#if HW_IP
conv2D(str_in, str_out, 0, pf);
#endif

    for (i=0; i<OWIDTH*OHEIGHT*FILTERS/PACKET/pf; i++) {
        tmpo = str_out.read();
        printf("%02d - result %d, %d, %d, %d\n", i, (int)tmpo.data.range(15,0), (int)tmpo.data.range(31,16),
                (int)tmpo.data.range(47,32), (int)tmpo.data.range(63,48));
    }


    return 0;
}
