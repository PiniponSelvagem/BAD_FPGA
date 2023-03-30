#include "test_simple.cpp"
#include "test_w_real_data.cpp"


int main() {

    //test_conv2D();
    test_maxpooling2D();

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
