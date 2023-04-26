#pragma once

#ifdef DEBUG
    #define PRINT_SEPARATOR                         printf("\n")
    #define PRINT_M_N_STRIDE(M, N, stride)          printf("M = %d | N = %d | stride = %d\n", M, N, stride)

    #define PRINT_UG_TITLE(N)                       printf("UPDATE GATE    : for i<N (N=%d)\n", N)
    #define PRINT_RG_TITLE(N)                       printf("RESET GATE     : for i<N (N=%d)\n", N)
    #define PRINT_CO_TITLE(N)                       printf("COMPUTE OUTPUT : for i<N (N=%d)\n", N)

    #define PRINT_K_LOOP(M)                         printf("    kernel : for j<M (M=%d)\n", M)
    #define PRINT_RK_LOOP(N)                        printf("    recurrent kernel : for j<N (N=%d)\n", N)

    #define PRINT_K(j, sum, w_idx, w, ip)           printf("        j = %d | sum = %.8f | weights[%2d] = %.8f | input[%2d] = %.8f\n", j, sum, w_idx, w, j, ip)
    #define PRINT_RK(j, sum, r_idx, r, s)           printf("        j = %d | sum = %.8f | r_weigh[%2d] = %.8f | state[%2d] = %.8f\n", j, sum, r_idx, r, j, s)

    #define PRINT_Z(i, sum, z)                      printf("  i = %d | sum = %.8f | z[%2d] = %.8f\n", i, sum, i, z)
    #define PRINT_R(i, sum, r)                      printf("  i = %d | sum = %.8f | r[%2d] = %.8f\n", i, sum, i, r)
    #define PRINT_H(i, sum, h)                      printf("  i = %d | sum = %.8f | h[%2d] = %.8f\n", i, sum, i, h)
#else
    #define PRINT_SEPARATOR
    #define PRINT_M_N_STRIDE(M, N, stride)

    #define PRINT_UG_TITLE(N)
    #define PRINT_RG_TITLE(N)
    #define PRINT_CO_TITLE(N)

    #define PRINT_K_LOOP(M)
    #define PRINT_RK_LOOP(N)
    
    #define PRINT_K(j, sum, w_idx, w, ip)
    #define PRINT_RK(j, sum, r_idx, r, s)

    #define PRINT_Z(i, sum, z)
    #define PRINT_R(i, sum, r)
    #define PRINT_H(i, sum, h)
#endif