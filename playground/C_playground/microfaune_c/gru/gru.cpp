#include "gru.h"

#include <stdio.h>
#include <string.h>

gru_t state[GRU_MAX_STATE][GRU_STATE_SIZE];     // 0 -> current, 1 -> next

void gru_clearState() {
    for (int i = 0; i < GRU_MAX_STATE; ++i) {
        for (int j = 0; j < GRU_STATE_SIZE; ++j) {
            state[i][j] = 0;
        }
    }
}

void gru_syncState() {
    for (int i = 0; i < GRU_STATE_SIZE; ++i) {
        state[0][i] = state[1][i];
    }
}
