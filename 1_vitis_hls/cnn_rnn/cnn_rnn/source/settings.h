#ifndef SETTINGS_H
#define SETTINGS_H



#define W_BIT_WIDTH 8
#define I_BIT_WIDTH 8
#define B_BIT_WIDTH 8

typedef ap_int<W_BIT_WIDTH> weigth_t;
typedef ap_int<I_BIT_WIDTH> imap_t;
typedef ap_int<I_BIT_WIDTH> omap_t;
typedef ap_int<B_BIT_WIDTH> bias_t;


#define MIN_VALUE(num_bits) ((-1LL << (num_bits - 1)))
#define MIN_VALUE_W         MIN_VALUE(W_BIT_WIDTH)
#define MIN_VALUE_I         MIN_VALUE(I_BIT_WIDTH)
#define MIN_VALUE_B         MIN_VALUE(B_BIT_WIDTH)

#define PADDING_SAME 2
#define C2D_OFFSET   (PADDING_SAME/2)


/* InputLayer */
#define IHEIGHT   1
#define IWIDTH    40
#define ICHANNELS 1


#define CNN_CHANNELS    64

/* Conv2D_1 */
#define C2D_1_IHEIGHT   (IHEIGHT + PADDING_SAME)
#define C2D_1_IWIDTH    (IWIDTH + PADDING_SAME)
#define C2D_1_ICHANNELS CNN_CHANNELS

#define C2D_1_KSIZE     3
#define C2D_1_BSIZE	    CNN_CHANNELS

#define C2D_1_OHEIGHT   C2D_1_IHEIGHT
#define C2D_1_OWIDTH    C2D_1_IWIDTH
#define C2D_1_OCHANNELS C2D_1_ICHANNELS


/* MaxPooling2D_1 */
#define MP2D_1_IHEIGHT   IHEIGHT
#define MP2D_1_IWIDTH    IWIDTH
#define MP2D_1_CHANNELS  1 /*CNN_CHANNELS*/  //TODO: change to 64 and test

#define MP2D_1_HSTRIDE   1
#define MP2D_1_WSTRIDE   2

#define MP2D_1_OHEIGHT   IHEIGHT
#define MP2D_1_OWIDTH    (IWIDTH/2)


/* TODO END */
#define OHEIGHT 1
#define OWIDTH  5


#endif