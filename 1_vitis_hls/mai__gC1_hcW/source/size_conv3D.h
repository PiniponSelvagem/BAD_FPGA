#ifndef SIZE_CONV3D
#define SIZE_CONV3D

#ifdef DEBUG_MODEL
#define IHEIGHT 3
#define IWIDTH  2
#else
#define IHEIGHT 	431
#define IWIDTH  	40
#define IWIDTH_1	20
#define IWIDTH_2	10
#endif

#define K_SIZE  3
#define OWIDTH  (IWIDTH-K_SIZE+1+2)		// hardcoded padding, equal do input
#define OHEIGHT (IHEIGHT-K_SIZE+1+2)	// hardcoded padding, equal to input


#endif // !SIZE_CONV3D
