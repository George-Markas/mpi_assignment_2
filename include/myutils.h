#ifndef MYUTILS_H
#define MYUTILS_H

int input_sanitizer(const char *buffer, int allow_negative, int allow_float);
int read_int(int *var, ...);
int read_float(float *var, ...);

#endif // MYUTILS_H

#ifdef MYUTILS_IMPLEMENTATION
#include <string.h>
#include <ctype.h>

/**
 * @brief Provides checks for non-numerics and negatives if validity_check_enabled is true.
 */
inline int input_sanitizer(const char *buffer, int allow_negative, int allow_float) {
    // In case of empty input, even if NULL macro is not defined
    if(buffer == NULL || *buffer == '\0')
        return 0;

    // Handle sign
    int i = 0;
    if(buffer[0] == '-') {
        if(allow_negative) {
            i++;
        } else {
            return 0;
        }
    }

    // In case input is just a newline
    if(buffer[i] == '\n')
        return 0;

    for(; buffer[i] != '\n'; i++) {
        #define FLOAT_CHECK
        if(allow_float)
            #define FLOAT_CHECK && (buffer[i] != 0x2e) // 0x2e = '.'

        if(!isdigit(buffer[i]) FLOAT_CHECK)
            return 0;
    }

    return 1;
}

#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#define BUFFER_SIZE 256

/**
 * @brief Reads an integer from stdin with optional checks for non-numerics and/or negatives.
 *
 * it read_int(int* var, int validity_check_enabled, int allow_negative)
 */
inline int read_int(int* var, ...) {
    va_list args;
    va_start(args, var);
    char buffer[BUFFER_SIZE];

    // If read fails, stop
    if(!fgets(buffer, BUFFER_SIZE, stdin))
        return EXIT_FAILURE;

    int validity_check_enabled = va_arg(args, int);
    if(validity_check_enabled) {
        int allow_negative = va_arg(args, int);
        char *str = "n";

        if (!allow_negative)
            str = " non-negative";

        va_end(args);

        while (!input_sanitizer(buffer, allow_negative, 0)) {
            fprintf(stderr, "\n\033[3;31mMust be a%s integer! Retry...\033[0m\n", str);
            fgets(buffer, BUFFER_SIZE, stdin);
        }
    }

    const int val = (int) strtol(buffer, NULL, 10);
    *var = val;
    return EXIT_SUCCESS;
}

/**
 * @brief Reads a float from stdin with optional checks for non-numerics and/or negatives.
 *
 * int read_float(float* var, int validity_check_enabled, int allow_negative)
 */
inline int read_float(float* var, ...) {
    va_list args;
    va_start(args, var);
    char buffer[BUFFER_SIZE];

    // If read fails, stop
    if(!fgets(buffer, BUFFER_SIZE, stdin))
        return EXIT_FAILURE;

    int validity_check_enabled = va_arg(args, int);
    if(validity_check_enabled) {
        int allow_negative = va_arg(args, int);
        char *str = " f";

        if (!allow_negative)
            str = " non-negative f";

        va_end(args);

        while (!input_sanitizer(buffer, allow_negative, 1)) {
            fprintf(stderr, "\n\033[3;31mMust be a%sloat! Retry...\033[0m\n", str);
            fgets(buffer, BUFFER_SIZE, stdin);
        }
    }

    const float val = strtof(buffer, NULL);
    *var = val;
    return EXIT_SUCCESS;
}

#endif // MYUTILS_IMPLEMENTATION