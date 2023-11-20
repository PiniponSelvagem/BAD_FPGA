#ifndef SIZE_CONV3D
#define SIZE_CONV3D


#define IHEIGHT 3	//431
#define IWIDTH  2	//40
#define IDEPTH  4   //64
#define K_SIZE  3
#define OWIDTH  (IWIDTH-K_SIZE+1+2)
#define OHEIGHT (IHEIGHT-K_SIZE+1+2)
#define UNROLL  4


#endif // !SIZE_CONV3D
