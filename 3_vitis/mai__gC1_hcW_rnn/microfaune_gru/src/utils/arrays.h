#ifndef ARRAYS_H
#define ARRAYS_H

/**
 * Flips array lines but not columns.
 *
 * Lets say we have a 2D array 3*3.
 * The array will be flip by lines and will keep the columns in the original order.
 * (numbers in the following array are the send order)
 * Original:
 * [
 *  [1 2 3]
 *  [4 5 6]
 *  [7 8 9]
 * ]
 * Result:
 * [
 *  [7 8 9]
 *  [4 5 6]
 *  [1 2 3]
 * ]
 */
void flipArrayLines(unsigned char* array, size_t size, int nCols) {
    size_t start = 0;
    size_t end = size - nCols;

    while (start < end) {
        // Swap elements at start and end indices for each row
        for (int i = 0; i < nCols; i++) {
            unsigned char temp = array[start + i];
            array[start + i] = array[end + i];
            array[end + i] = temp;
        }

        // Move indices towards the center for the next row
        start += nCols;
        end -= nCols;
    }
}


/**
 * Saves the result of the Bidirectional GRU "input", to the "output" array.
 * The result of a Bidirectional GRU has its lines divided into 2 sections, the forward and the backward.
 * The forward section is from the start of the line to the middle of the line, and the backward section the 2nd half.
 * Example, with 2 GRU Cells and 3 lines: (F -> Forward, B -> Backward)
 * [
 * 	[ F, F, B, B ],		--> 1st line
 * 	[ F, F, B, B ],		--> 2nd line
 * 	[ F, F, B, B ]		--> 3rd line
 * ]
 * Since the developed Bidirectional GRU only outputs 1 direction at a time, this function will help to place the output
 * at the correct places. When "direction=1" will place all the "input" into the "F" location, "direction=0" on the "B"
 * location.
 *
 * input: the result from the Bidirectional GRU
 * inSize: size of the result from the BiGRU and ALWAYS half of the size of the output array
 * direction: 1 == Forward, 0 == Backward
 * gruCells: number of cells of the GRU, also equal to the number of columns the result has
 */
void saveGruOutputTo(unsigned char* input, size_t inSize, unsigned char* output, int direction, int gruCells) {
    size_t inputIndex = 0;
    size_t outputIndex;

    if (direction == 1) {
        outputIndex = 0;
    } else {
        outputIndex = 2*inSize - gruCells;
    }

    while (inputIndex < inSize) {
    	for (int i = 0; i < gruCells; ++i) {
			if (direction == 1) {
				output[outputIndex] = input[inputIndex];
			} else {
				output[outputIndex] = input[inputIndex];
			}
			++inputIndex;
			++outputIndex;
		}

    	if (direction == 1) {
    		outputIndex += gruCells;
		} else {
			outputIndex -= (2 * gruCells) + gruCells;
		}
    }
}

#endif // ARRAYS_H
