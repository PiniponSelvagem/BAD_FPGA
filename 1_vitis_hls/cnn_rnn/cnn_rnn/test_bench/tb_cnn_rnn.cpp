//#define TEST_WITH_SIMPLE_DATA

#ifdef  TEST_WITH_SIMPLE_DATA
    #include "test_simple.cpp"
#else
    #include "test_w_real_data.cpp"
#endif

#include "gru.cpp"

#define OUTPUT_SIZE     128
float output[OUTPUT_SIZE];

int main() {
	test_gru(output);

	for (int i = 0; i < OUTPUT_SIZE; ++i) {
		printf("%3d > %.16f\n", i, output[i]);
	}

    //test_conv2D();
    //test_maxpooling2D();

	/*
	printf("\nOutput Image\n");
	for (i = 0; i < OHEIGHT; i++) {
		for (j = 0; j < OWIDTH; j++) {
			printf("%4d ", img_out[i * OWIDTH + j]);
		}
		printf("\n");
	}

    /*
	for (err_cnt = 0, i = 0; i < OHEIGHT; i++) {
		for (j = 0; j < OWIDTH; j++) {
			if (hw_img_out[i * OWIDTH + j] != sw_img_out[i * OWIDTH + j]) {
				err_cnt++;
				printf("%d,%d: %d != %d\n", i, j, hw_img_out[i * OWIDTH + j],
						sw_img_out[i * OWIDTH + j]);
			}
		}
	}
	
    return err_cnt;
    */

    return 0;   // TODO
}
