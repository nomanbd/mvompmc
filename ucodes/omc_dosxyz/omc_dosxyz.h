#ifndef OMC_DOSXYZ_H
#define OMC_DOSXYZ_H

#include <signal.h>

/* Library entry point exposed when omc_dosxyz.c is compiled with
 * -DOMC_DOSXYZ_AS_LIBRARY. The signature mirrors the standalone
 * binary's main() — argv[1..] should contain the same flags
 * (-i input_base, -o output_name) the executable accepts.
 *
 * Returns the same exit code as the binary (0 on success). The
 * function still reads its input files from disk and writes the
 * .3ddose text output; the binary dose-handoff lives in
 * omc_dosxyz_dump_dose below. */
int omc_dosxyz_main(int argc, char **argv);

/* Serialise the post-simulation dose grid to a file descriptor in
 * the "QDD1" binary format (see omc_dosxyz.c for a precise
 * description). Intended to be called once, after omc_dosxyz_main
 * returns 0, in the same process that ran the simulation. The
 * caller is responsible for closing fd.
 *
 * Returns 0 on success, -1 on a short write. */
int omc_dosxyz_dump_dose(int fd);

/* Cooperative-cancellation flag (slice 5c). The kernel checks this
 * at every batch boundary; setting it to 1 causes the next iteration
 * to break out cleanly, finalize whatever batches have completed,
 * and return through the normal cleanup path.
 *
 * Reset to 0 at the start of every omc_dosxyz_main call.
 *
 * On Linux/macOS the kernel installs a SIGINT handler that writes 1
 * here, so a Ctrl+C in a terminal triggers cancellation. Hosting
 * processes can also write directly. */
extern volatile sig_atomic_t omc_dosxyz_cancel;

#endif
