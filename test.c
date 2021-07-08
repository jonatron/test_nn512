#define _GNU_SOURCE
#include <pthread.h>
#include <sched.h>
#include <stdlib.h>
#include <assert.h>
#include <stdio.h>
#include <time.h>
#include "ResNet50.h"

// gcc -c -std=c99 -pthread -Ofast -mavx512f ResNet50.c
// gcc -std=c99 -pthread -Ofast -mavx512f  ResNet50.o test.c -o test


int main() {
    char* err;

    int threads = 1;

    srand(1);

    ResNet50Params* params = malloc(sizeof(ResNet50Params));

    printf("reading %lu %lu \n", sizeof(ResNet50Params), sizeof(ResNet50Params) / sizeof(float));
    FILE* f = fopen("float.dat", "r");
    int read = fread(params, sizeof(float), sizeof(ResNet50Params) / sizeof(float), f);
    printf("read %i params \n", read);

    // first floats: tf.Tensor([ 0.04097798 -0.0263161  -1.3460633 ]
    // printf("first floats %f %f \n", params->bn10Means[0], params->bn10Means[1]);

    ResNet50Net* net;
    err = ResNet50NetCreate(&net, params, threads);
    free(params);

    ResNet50Engine* engine0;
    err = ResNet50EngineCreate(&engine0, net, threads);

    if (err) { // Nonzero err means failure; engine is unmodified.
        printf("%s\n", err); // Explain the failure, add a newline.
        free(err); // Free the error string to avoid a memory leak.
        ResNet50EngineDestroy(engine0); // Terminate threads, free memory.
        exit(1); // Exit, or propagate the failure some other way.
    }

    float* imageData = malloc(sizeof(float)*3*224*224);
    float* probData = malloc(sizeof(float)*1000);

    FILE* f2 = fopen("elephant.dat", "r");
    printf("reading elephant \n");
    int read2 = fread(imageData, sizeof(float), 3*224*224, f2);
    printf("read %i elephant items \n", read2);

    for(int i = 0; i < 1; i++) {
        clock_t begin = clock();
        ResNet50EngineInference( // This function cannot fail.
            engine0, // Pass an Engine as the first argument.
            imageData, // The tensor arguments are sorted by name.
            probData
        );

        clock_t end = clock();
        double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
        printf("Time spent %f\n", time_spent);

        for(int j = 0; j < 1000; j++) {
            if(probData[j] > 0.01) {
                printf("idx: %d prob: %.20f \n", j, probData[j]);
            }
        }
    }

    free(imageData);
    free(probData);

}
