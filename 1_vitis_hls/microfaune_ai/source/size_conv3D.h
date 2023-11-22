#ifndef SIZE_CONV3D
#define SIZE_CONV3D


#define IHEIGHT 3	//431
#define IWIDTH  2	//40
#define IDEPTH  CHANNELS  //64 ; cannot be lower than PACKET because of "IDEPTH/PACKET" in axis_conv3D
#define K_SIZE  3
#define OWIDTH  (IWIDTH-K_SIZE+1+2)		// hardcoded padding, equal do input
#define OHEIGHT (IHEIGHT-K_SIZE+1+2)	// hardcoded padding, equal to input
#define UNROLL  4


#endif // !SIZE_CONV3D
