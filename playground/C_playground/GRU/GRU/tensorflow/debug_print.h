#pragma once

#ifdef DEBUG_PRINT
    #include <stdio.h>
    #define PRINT_SEPARATOR      printf("\n")
    #define PRINT_STRING(str)    printf("%s\n", str);
    #define PRINT_ARRAY_1D_X(title, arr, size, format) \
        do { \
            printf("%s = [ ", title); \
            for (int i = 0; i < size; i++) { \
                printf(format, arr[i]); \
            } \
            printf("]\n"); \
        } while(0)
    #define PRINT_ARRAY_1D(title, arr, size)    PRINT_ARRAY_1D_X(title, arr, size, "%.10f ")
    #define PRINT_ARRAY_2D_X(title, arr, rows, cols, format) \
        do { \
            printf("%s = [", title); \
            for (int i = 0; i < rows; i++) { \
                printf(" [ "); \
                for (int j = 0; j < cols; j++) { \
                    printf(format, arr[i][j]); \
                } \
                printf("]"); \
                if (i != rows - 1) { \
                    printf(" "); \
                } \
                if ((i + 1) % rows == 0 && i > rows) { \
                    printf("\n"); \
                } \
            } \
            printf(" ]\n"); \
        } while(0)
    #define PRINT_ARRAY_2D(title, arr, rows, cols)    PRINT_ARRAY_2D_X(title, arr, rows, cols, "%.10f ")
#else
    #define PRINT_SEPARATOR ;
    #define PRINT_STRING(str) ;
    #define PRINT_ARRAY_1D_X(title, arr, size, format) ;
    #define PRINT_ARRAY_1D(title, arr, size) ;
    #define PRINT_ARRAY_2D_X(title, arr, rows, cols, format) ;
    #define PRINT_ARRAY_2D(title, arr, rows, cols) ;
#endif