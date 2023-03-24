#define W_BIT_WIDTH 8
#define I_BIT_WIDTH 8
#define B_BIT_WIDTH 8

#define PADDING_SAME 2
#define C2D_OFFSET   (PADDING_SAME/2)


/* InputLayer */
#define IHEIGHT   1
#define IWIDTH    40
#define ICHANNELS 1


/* Conv2D_1 */
#define IC2D_1_IHEIGHT   (IHEIGHT + PADDING_SAME)
#define IC2D_1_IWIDTH    (IWIDTH + PADDING_SAME)
#define IC2D_1_ICHANNELS 64

#define IC2D_1_KSIZE     3
#define IC2D_1_BSIZE	 64

#define IC2D_1_OHEIGHT   IC2D_1_IHEIGHT
#define IC2D_1_OWIDTH    IC2D_1_IWIDTH
#define IC2D_1_OCHANNELS 64



/* TODO END */
#define OHEIGHT 1
#define OWIDTH  5
