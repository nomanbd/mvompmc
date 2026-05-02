#ifndef OMC_DOSXYZ_H
#define OMC_DOSXYZ_H

/* Library entry point exposed when omc_dosxyz.c is compiled with
 * -DOMC_DOSXYZ_AS_LIBRARY. The signature mirrors the standalone
 * binary's main() — argv[1..] should contain the same flags
 * (-i input_base, -o output_name) the executable accepts.
 *
 * Returns the same exit code as the binary (0 on success). The
 * function reads its inputs from disk and writes the output .3ddose
 * file at the configured path; in-memory I/O is a future slice. */
int omc_dosxyz_main(int argc, char **argv);

#endif
