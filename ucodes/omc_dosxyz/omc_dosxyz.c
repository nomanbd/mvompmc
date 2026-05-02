/******************************************************************************
 ompMC - An OpenMP parallel implementation for Monte Carlo particle transport
 simulations
 
 Copyright (C) 2018 Edgardo Doerner (edoerner@fis.puc.cl)


 This program is free software: you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation, either version 3 of the License, or
 (at your option) any later version.

 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.

 You should have received a copy of the GNU General Public License
 along with this program.  If not, see <https://www.gnu.org/licenses/>.
*****************************************************************************/

/******************************************************************************
 omc_dosxyz - An ompMC user code to calculate deposited dose on voxelized 
 geometries.  
*****************************************************************************/

#include <ctype.h>
#include <math.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#ifdef _OPENMP
    #include <omp.h>
#endif

#include "omc_utilities.h"
#include "ompmc.h"
#include "omc_random.h"

/******************************************************************************/
/* Parsing program options with getopt long
 http://www.gnu.org/software/libc/manual/html_node/Getopt.html#Getopt */
#include <getopt.h>

/******************************************************************************/
/* Geometry definitions */
struct Geom {
    int *med_indices;           // index of the media in each voxel
    double *med_densities;      // density of the medium in each voxel
    
    int isize;                  // number of voxels on each direction
    int jsize;
    int ksize;
    
    double *xbounds;            // boundaries of voxels on each direction
    double *ybounds;
    double *zbounds;
};
struct Geom geometry;

void initPhantom() {
    
    /* Get phantom file path from input data */
    char phantom_file[128];
    char buffer[BUFFER_SIZE];
    
    if (getInputValue(buffer, "phantom file") != 1) {
        printf("Can not find 'phantom file' key on input file.\n");
        exit(EXIT_FAILURE);
    }
    removeSpaces(phantom_file, buffer);
    
    /* Open .egsphant file */
    FILE *fp;
    
    if ((fp = fopen(phantom_file, "r")) == NULL) {
        printf("Unable to open file: %s\n", phantom_file);
        exit(EXIT_FAILURE);
    }
    
    printf("Path to phantom file : %s\n", phantom_file);
    
    /* Get number of media in the phantom */
    fgets(buffer, BUFFER_SIZE, fp);
    media.nmed = atoi(buffer);
    
    /* Get media names on phantom file */
    for (int i=0; i<media.nmed; i++) {
        fgets(buffer, BUFFER_SIZE, fp);
        removeSpaces(media.med_names[i], buffer);
    }
    
    /* Skip next line, it contains dummy input */
    fgets(buffer, BUFFER_SIZE, fp);
    
    /* Read voxel numbers on each direction */
    fgets(buffer, BUFFER_SIZE, fp);
    sscanf(buffer, "%d %d %d", &geometry.isize,
           &geometry.jsize, &geometry.ksize);
    
    /* Read voxel boundaries on each direction */
    geometry.xbounds = malloc((geometry.isize + 1)*sizeof(double));
    geometry.ybounds = malloc((geometry.jsize + 1)*sizeof(double));
    geometry.zbounds = malloc((geometry.ksize + 1)*sizeof(double));
    
    for (int i=0; i<=geometry.isize; i++) {
        fscanf(fp, "%lf", &geometry.xbounds[i]);
    }
    for (int i=0; i<=geometry.jsize; i++) {
        fscanf(fp, "%lf", &geometry.ybounds[i]);
     }
    for (int i=0; i<=geometry.ksize; i++) {
        fscanf(fp, "%lf", &geometry.zbounds[i]);
    }
    
    /* Skip the rest of the last line read before */
    fgets(buffer, BUFFER_SIZE, fp);
    
    /* Read media indices */
    int irl = 0;    // region index
    char idx;
    geometry.med_indices =
        malloc(geometry.isize*geometry.jsize*geometry.ksize*sizeof(int));
    for (int k=0; k<geometry.ksize; k++) {
        for (int j=0; j<geometry.jsize; j++) {
            for (int i=0; i<geometry.isize; i++) {
                irl = i + j*geometry.isize + k*geometry.jsize*geometry.isize;
                idx = fgetc(fp);
                /* Convert digit stored as char to int */
                geometry.med_indices[irl] = idx - '0';
            }
            /* Jump to next line */
            fgets(buffer, BUFFER_SIZE, fp);
        }
        /* Skip blank line */
        fgets(buffer, BUFFER_SIZE, fp);
    }
    
    /* Read media densities */
    geometry.med_densities =
        malloc(geometry.isize*geometry.jsize*geometry.ksize*sizeof(double));
    for (int k=0; k<geometry.ksize; k++) {
        for (int j=0; j<geometry.jsize; j++) {
            for (int i=0; i<geometry.isize; i++) {
                irl = i + j*geometry.isize + k*geometry.jsize*geometry.isize;
                fscanf(fp, "%lf", &geometry.med_densities[irl]);
            }
        }
        /* Skip blank line */
        fgets(buffer, BUFFER_SIZE, fp);
    }
    
    /* Summary with geometry information */
    printf("Number of media in phantom : %d\n", media.nmed);
    printf("Media names: ");
    for (int i=0; i<media.nmed; i++) {
        printf("%s, ", media.med_names[i]);
    }
    printf("\n");
    printf("Number of voxels on each direction (X,Y,Z) : (%d, %d, %d)\n",
           geometry.isize, geometry.jsize, geometry.ksize);
    printf("Minimum and maximum boundaries on each direction : \n");
    printf("\tX (cm) : %lf, %lf\n",
           geometry.xbounds[0], geometry.xbounds[geometry.isize]);
    printf("\tY (cm) : %lf, %lf\n",
           geometry.ybounds[0], geometry.ybounds[geometry.jsize]);
    printf("\tZ (cm) : %lf, %lf\n",
           geometry.zbounds[0], geometry.zbounds[geometry.ksize]);
    
    /* Close phantom file */
    fclose(fp);
    
    return;
}

void cleanPhantom() {
    
    free(geometry.xbounds);
    free(geometry.ybounds);
    free(geometry.zbounds);
    free(geometry.med_indices);
    free(geometry.med_densities);
    return;
}

void howfar(int *idisc, int *irnew, double *ustep) {
    
    int np = stack.np;
    int irl = stack.ir[np];
    double dist = 0.0;
    
    if (stack.ir[np] == 0) {
        /* The particle is outside the geometry, terminate history */
        *idisc = 1;
        return;
    }
    
    /* If here, the particle is in the geometry, do transport checks */
    int ijmax = geometry.isize*geometry.jsize;
    int imax = geometry.isize;
    
    /* First we need to decode the region number of the particle in terms of
     the region indices in each direction */
    int irx = (irl - 1)%imax;
    int irz = (irl - 1 - irx)/ijmax;
    int iry = ((irl - 1 - irx) - irz*ijmax)/imax;
    
    /* Check in z-direction */
    if (stack.w[np] > 0.0) {
        /* Going towards outer plane */
        dist = (geometry.zbounds[irz+1] - stack.z[np])/stack.w[np];
        if (dist < *ustep) {
            *ustep = dist;
            if (irz != (geometry.ksize - 1)) {
                *irnew = irl + ijmax;
            }
            else {
                *irnew = 0; /* leaving geometry */
            }
        }
    }
    
    else if (stack.w[np] < 0.0) {
        /* Going towards inner plane */
        dist = -(stack.z[np] - geometry.zbounds[irz])/stack.w[np];
        if (dist < *ustep) {
            *ustep = dist;
            if (irz != 0) {
                *irnew = irl - ijmax;
            }
            else {
                *irnew = 0; /* leaving geometry */
            }
        }
    }

    /* Check in x-direction */
    if (stack.u[np] > 0.0) {
        /* Going towards positive plane */
        dist = (geometry.xbounds[irx+1] - stack.x[np])/stack.u[np];
        if (dist < *ustep) {
            *ustep = dist;
            if (irx != (geometry.isize - 1)) {
                *irnew = irl + 1;
            }
            else {
                *irnew = 0; /* leaving geometry */
            }
        }
    }
    
    else if (stack.u[np] < 0.0) {
        /* Going towards negative plane */
        dist = -(stack.x[np] - geometry.xbounds[irx])/stack.u[np];
        if (dist < *ustep) {
            *ustep = dist;
            if (irx != 0) {
                *irnew = irl - 1;
            }
            else {
                *irnew = 0; /* leaving geometry */
            }
        }
    }
    
    /* Check in y-direction */
    if (stack.v[np] > 0.0) {
        /* Going towards positive plane */
        dist = (geometry.ybounds[iry+1] - stack.y[np])/stack.v[np];
        if (dist < *ustep) {
            *ustep = dist;
            if (iry != (geometry.jsize - 1)) {
                *irnew = irl + imax;
            }
            else {
                *irnew = 0; /* leaving geometry */
            }
        }
    }
    
    else if (stack.v[np] < 0.0) {
        /* Going towards negative plane */
        dist = -(stack.y[np] - geometry.ybounds[iry])/stack.v[np];
        if (dist < *ustep) {
            *ustep = dist;
            if (iry != 0) {
                *irnew = irl - imax;
            }
            else {
                *irnew = 0; /* leaving geometry */
            }
        }
    }
    
    return;
}

double hownear(void) {
    
    int np = stack.np;
    int irl = stack.ir[np];
    double tperp = 1.0E10;  /* perpendicular distance to closest boundary */
    
    if (irl == 0) {
        /* Particle exiting geometry */
        tperp = 0.0;
    }
    else {
        /* In the geometry, do transport checks */
        int ijmax = geometry.isize*geometry.jsize;
        int imax = geometry.isize;
        
        /* First we need to decode the region number of the particle in terms
         of the region indices in each direction */
        int irx = (irl - 1)%imax;
        int irz = (irl - 1 - irx)/ijmax;
        int iry = ((irl - 1 - irx) - irz*ijmax)/imax;
        
        /* Check in x-direction */
        tperp = fmin(tperp, geometry.xbounds[irx+1] - stack.x[np]);
        tperp = fmin(tperp, stack.x[np] - geometry.xbounds[irx]);
        
        /* Check in y-direction */
        tperp = fmin(tperp, geometry.ybounds[iry+1] - stack.y[np]);
        tperp = fmin(tperp, stack.y[np] - geometry.ybounds[iry]);
        
        /* Check in z-direction */
        tperp = fmin(tperp, geometry.zbounds[irz+1] - stack.z[np]);
        tperp = fmin(tperp, stack.z[np] - geometry.zbounds[irz]);
    }
    
    return tperp;
}
/******************************************************************************/

/******************************************************************************/
/* Source definitions */
const int MXEBIN = 200;     // number of energy bins of spectrum
const int INVDIM = 1000;    // number of bins in inverse CDF

struct Source {
    int spectrum;               // 0 : monoenergetic, 1 : spectrum
    int charge;                 // 0 : photons, -1 : electron, +1 : positron
    
    /* For monoenergetic source */
    double energy;
    
    /* For spectrum */
    double deltak;              // number of elements in inverse CDF
    double *cdfinv1;            // energy value of bin
    double *cdfinv2;            // prob. that particle has energy xi
    
    /* Source shape information */
    double ssd;                 // distance of point source to phantom surface
    double xinl, xinu;          // lower and upper x-bounds of the field on
                                // phantom surface
    double yinl, yinu;          // lower and upper y-bounds of the field on
                                // phantom surface
    double xsize, ysize;        // x- and y-width of collimated field
    int ixinl, ixinu;        // lower and upper x-bounds indices of the
                                // field on phantom surface
    int iyinl, iyinu;        // lower and upper y-bounds indices of the
                                // field on phantom surface

    /* Rotated-beam mode (qdc Phase 2 slice 2e). When rotated_beam == 0,
     * the legacy ssd-based geometry above is used unchanged. When 1,
     * the source can sit at an arbitrary 3D position and the beam axis
     * can point in any direction; particles are injected from
     * source_pos toward a sample point on the iso plane and ray-box
     * intersected with the phantom AABB to find the entry voxel.
     *
     * In rotated_beam mode, xinl/xinu/yinl/yinu are reinterpreted as
     * collimator bounds in the iso plane (perpendicular to beam_axis),
     * not on the phantom surface. Distances are cm.
     *
     * cright and cup are unit vectors spanning the iso plane:
     *   target = iso + px*cright + py*cup     for px,py in [xinl,xinu]
     *                                              x [yinl,yinu]
     */
    int rotated_beam;
    double source_pos[3];     /* cm, IEC frame */
    double beam_axis[3];      /* unit vector from source toward iso */
    double iso[3];            /* cm, IEC frame */
    double cright[3];         /* in-beam +x basis (unit, perp beam_axis) */
    double cup[3];            /* in-beam +y basis (unit, perp beam_axis) */

    /* IAEA phase-space mode (qdc Phase 2 slice 2f). When phsp_active
     * is set, particle properties (type / energy / position-in-iso-
     * plane / direction / weight) come from a phsp file rather than
     * the analytic spectrum sampler. The same rotated-beam basis
     * (iso, cright, cup, beam_axis) transforms phsp coordinates into
     * the MVompMC voxel-grid frame; the same ray-box intersection
     * locates the phantom entry voxel.
     *
     * Supports the canonical 32-byte little-endian record described
     * in qdc's docs/phase-space.md. */
    int     phsp_active;
    FILE   *phsp_fp;
    long long phsp_total_records;
    long long phsp_records_read;
};
struct Source source;

void initSource() {
    
    /* Get spectrum file path from input data */
    char spectrum_file[128];
    char buffer[BUFFER_SIZE];
    
    source.spectrum = 1;    /* energy spectrum as default case */
    
    /* First check of spectrum file was given as an input */
    if (getInputValue(buffer, "spectrum file") != 1) {
        printf("Can not find 'spectrum file' key on input file.\n");
        printf("Switch to monoenergetic case.\n");
        source.spectrum = 0;    /* monoenergetic source */
    }
    
    if (source.spectrum) {
        removeSpaces(spectrum_file, buffer);
        
        /* Open .source file */
        FILE *fp;
        
        if ((fp = fopen(spectrum_file, "r")) == NULL) {
            printf("Unable to open file: %s\n", spectrum_file);
            exit(EXIT_FAILURE);
        }
        
        printf("Path to spectrum file : %s\n", spectrum_file);
        
        /* Read spectrum file title */
        fgets(buffer, BUFFER_SIZE, fp);
        printf("Spectrum file title: %s", buffer);
        
        /* Read number of bins and spectrum type */
        double enmin;   /* lower energy of first bin */
        int nensrc;     /* number of energy bins in spectrum histogram */
        int imode;      /* 0 : histogram counts/bin, 1 : counts/MeV*/
        
        fgets(buffer, BUFFER_SIZE, fp);
        sscanf(buffer, "%d %lf %d", &nensrc, &enmin, &imode);
        
        if (nensrc > MXEBIN) {
            printf("Number of energy bins = %d is greater than max allowed = "
                   "%d. Increase MXEBIN macro!\n", nensrc, MXEBIN);
            exit(EXIT_FAILURE);
        }
        
        /* upper energy of bin i in MeV */
        double *ensrcd = malloc(nensrc*sizeof(double));
        /* prob. of finding a particle in bin i */
        double *srcpdf = malloc(nensrc*sizeof(double));
        
        /* Read spectrum information */
        for (int i=0; i<nensrc; i++) {
            fgets(buffer, BUFFER_SIZE, fp);
            sscanf(buffer, "%lf %lf", &ensrcd[i], &srcpdf[i]);
        }
        printf("Have read %d input energy bins from spectrum file.\n", nensrc);
        
        if (imode == 0) {
            printf("Counts/bin assumed.\n");
        }
        else if (imode == 1) {
            printf("Counts/MeV assumed.\n");
            srcpdf[0] *= (ensrcd[0] - enmin);
            for(int i=1; i<nensrc; i++) {
                srcpdf[i] *= (ensrcd[i] - ensrcd[i - 1]);
            }
        }
        else {
            printf("Invalid mode number in spectrum file.");
            exit(EXIT_FAILURE);
        }
        
        double ein = ensrcd[nensrc - 1];
        printf("Energy ranges from %f to %f MeV\n", enmin, ein);
        
        /* Initialization routine to calculate the inverse of the
         cumulative probability distribution that is used during execution to
         sample the incident particle energy. */
        double *srccdf = malloc(nensrc*sizeof(double));
        
        srccdf[0] = srcpdf[0];
        for (int i=1; i<nensrc; i++) {
            srccdf[i] = srccdf[i-1] + srcpdf[i];
        }
        
        double fnorm = 1.0/srccdf[nensrc - 1];
        double binsok = 0.0;
        source.deltak = INVDIM; /* number of elements in inverse CDF */
        double gridsz = 1.0f/source.deltak;
        
        for (int i=0; i<nensrc; i++) {
            srccdf[i] *= fnorm;
            if (i == 0) {
                if (srccdf[0] <= 3.0*gridsz) {
                    binsok = 1.0;
                }
            }
            else {
                if ((srccdf[i] - srccdf[i - 1]) < 3.0*gridsz) {
                    binsok = 1.0;
                }
            }
        }
        
        if (binsok != 0.0) {
            printf("Warning!, some of normalized bin probabilities are "
                   "so small that bins may be missed.\n");
        }

        /* Calculate cdfinv. This array allows the rapid sampling for the
         energy by precomputing the results for a fine grid. */
        source.cdfinv1 = malloc(source.deltak*sizeof(double));
        source.cdfinv2 = malloc(source.deltak*sizeof(double));
        double ak;
        
        for (int k=0; k<source.deltak; k++) {
            ak = (double)k*gridsz;
            int i;
            
            for (i=0; i<nensrc; i++) {
                if (ak <= srccdf[i]) {
                    break;
                }
            }
            
            /* We should fall here only through the above break sentence. */
            if (i != 0) {
                source.cdfinv1[k] = ensrcd[i - 1];
            }
            else {
                source.cdfinv1[k] = enmin;
            }
            source.cdfinv2[k] = ensrcd[i] - source.cdfinv1[k];
            
        }
        
        /* Cleaning */
        fclose(fp);
        free(ensrcd);
        free(srcpdf);
        free(srccdf);
    }
    else {  /* monoenergetic source */
        if (getInputValue(buffer, "mono energy") != 1) {
            printf("Can not find 'mono energy' key on input file.\n");
            exit(EXIT_FAILURE);
        }
        source.energy = atof(buffer);
        printf("%f monoenergetic source\n", source.energy);
        
    }
    
    /* Initialize geometrical data of the source */
    
    /* Read collimator rectangle */
    if (getInputValue(buffer, "collimator bounds") != 1) {
        printf("Can not find 'collimator bounds' key on input file.\n");
        exit(EXIT_FAILURE);
    }
    sscanf(buffer, "%lf %lf %lf %lf", &source.xinl,
           &source.xinu, &source.yinl, &source.yinu);
    
    /* Calculate x-direction input zones */
    if (source.xinl < geometry.xbounds[0]) {
        source.xinl = geometry.xbounds[0];
    }
    if (source.xinu <= source.xinl) {
        source.xinu = source.xinl;  /* default a pencil beam */
    }
    
    /* Check radiation field is not too big against the phantom */
    if (source.xinu > geometry.xbounds[geometry.isize]) {
        source.xinu = geometry.xbounds[geometry.isize];
    }
    if (source.xinl > geometry.xbounds[geometry.isize]) {
        source.xinl = geometry.xbounds[geometry.isize];
    }
    
    /* Now search for initial region x index range */
    printf("Index ranges for radiation field:\n");
    source.ixinl = 0;
    while ((geometry.xbounds[source.ixinl] <= source.xinl) &&
           (geometry.xbounds[source.ixinl + 1] < source.xinl)) {
        source.ixinl++;
    }
        
    source.ixinu = source.ixinl - 1;
    while ((geometry.xbounds[source.ixinu] <= source.xinu) &&
           (geometry.xbounds[source.ixinu + 1] < source.xinu)) {
        source.ixinu++;
    }
    printf("i index ranges over i = %d to %d\n", source.ixinl, source.ixinu);
    
    /* Calculate y-direction input zones */
    if (source.yinl < geometry.ybounds[0]) {
        source.yinl = geometry.ybounds[0];
    }
    if (source.yinu <= source.yinl) {
        source.yinu = source.yinl;  /* default a pencil beam */
    }
    
    /* Check radiation field is not too big against the phantom */
    if (source.yinu > geometry.ybounds[geometry.jsize]) {
        source.yinu = geometry.ybounds[geometry.jsize];
    }
    if (source.yinl > geometry.ybounds[geometry.jsize]) {
        source.yinl = geometry.ybounds[geometry.jsize];
    }
    
    /* Now search for initial region y index range */
    source.iyinl = 0;
    while ((geometry.ybounds[source.iyinl] <= source.yinl) &&
           (geometry.ybounds[source.iyinl + 1] < source.yinl)) {
        source.iyinl++;
    }
    source.iyinu = source.iyinl - 1;
    while ((geometry.ybounds[source.iyinu] <= source.yinu) &&
           (geometry.ybounds[source.iyinu + 1] < source.yinu)) {
        source.iyinu++;
    }
    printf("j index ranges over i = %d to %d\n", source.iyinl, source.iyinu);

    /* Calculate collimator sizes */
    source.xsize = source.xinu - source.xinl;
    source.ysize = source.yinu - source.yinl;
    
    /* Read source charge */
    if (getInputValue(buffer, "charge") != 1) {
        printf("Can not find 'charge' key on input file.\n");
        exit(EXIT_FAILURE);
    }
    
    source.charge = atoi(buffer);
    if (source.charge < -1 || source.charge > 1) {
        printf("Particle kind not recognized.\n");
        exit(EXIT_FAILURE);
    }
    
    /* Read source SSD */
    if (getInputValue(buffer, "ssd") != 1) {
        printf("Can not find 'ssd' key on input file.\n");
        exit(EXIT_FAILURE);
    }
    
    source.ssd = atof(buffer);
    if (source.ssd < 0) {
        printf("SSD must be greater than zero.\n");
        exit(EXIT_FAILURE);
    }
    
    /* Print some information for debugging purposes */
    if (verbose_flag) {
        printf("Source information :\n");
        printf("\t Charge = %d\n", source.charge);
        printf("\t SSD (cm) = %f\n", source.ssd);
        printf("Collimator :\n");
        printf("\t x (cm) : min = %f, max = %f\n", source.xinl, source.xinu);
        printf("\t y (cm) : min = %f, max = %f\n", source.yinl, source.yinu);
        printf("Sizes :\n");
        printf("\t x (cm) = %f, y (cm) = %f\n", source.xsize, source.ysize);
    }

    /* Slice 2f phase-space mode is keyed off `source type = phase_space`
     * + a `phsp file = <path>` pointing at the IAEAheader. Initialised
     * lazily below; rotated-beam basis (set after this) is shared. */
    source.phsp_active = 0;
    source.phsp_fp = NULL;
    source.phsp_total_records = 0;
    source.phsp_records_read = 0;
    char source_type_buf[BUFFER_SIZE];
    int phsp_requested = 0;
    if (getInputValue(source_type_buf, "source type") == 1) {
        if (strcmp(source_type_buf, "phase_space") == 0) {
            phsp_requested = 1;
        }
    }

    /* Optional rotated-beam mode (qdc Phase 2 slice 2e). Activated if a
     * `source position` key is present. All vectors are in the same IEC
     * frame as the phantom voxel grid. */
    source.rotated_beam = 0;
    if (getInputValue(buffer, "source position") == 1) {
        if (sscanf(buffer, "%lf %lf %lf",
                   &source.source_pos[0],
                   &source.source_pos[1],
                   &source.source_pos[2]) != 3) {
            printf("Malformed 'source position'; expected 'sx sy sz'.\n");
            exit(EXIT_FAILURE);
        }
        if (getInputValue(buffer, "isocenter") != 1 ||
            sscanf(buffer, "%lf %lf %lf",
                   &source.iso[0], &source.iso[1], &source.iso[2]) != 3) {
            printf("rotated_beam: missing 'isocenter = ix iy iz'.\n");
            exit(EXIT_FAILURE);
        }
        if (getInputValue(buffer, "collimator right") != 1 ||
            sscanf(buffer, "%lf %lf %lf",
                   &source.cright[0], &source.cright[1], &source.cright[2]) != 3) {
            printf("rotated_beam: missing 'collimator right = rx ry rz'.\n");
            exit(EXIT_FAILURE);
        }
        if (getInputValue(buffer, "collimator up") != 1 ||
            sscanf(buffer, "%lf %lf %lf",
                   &source.cup[0], &source.cup[1], &source.cup[2]) != 3) {
            printf("rotated_beam: missing 'collimator up = ux uy uz'.\n");
            exit(EXIT_FAILURE);
        }
        double bx = source.iso[0] - source.source_pos[0];
        double by = source.iso[1] - source.source_pos[1];
        double bz = source.iso[2] - source.source_pos[2];
        double bn = sqrt(bx*bx + by*by + bz*bz);
        if (bn <= 0.0) {
            printf("rotated_beam: source and isocenter coincide.\n");
            exit(EXIT_FAILURE);
        }
        source.beam_axis[0] = bx / bn;
        source.beam_axis[1] = by / bn;
        source.beam_axis[2] = bz / bn;
        source.rotated_beam = 1;
        if (verbose_flag) {
            printf("Rotated-beam mode enabled.\n");
            printf("\t source = (%.4f, %.4f, %.4f)\n",
                   source.source_pos[0], source.source_pos[1], source.source_pos[2]);
            printf("\t iso    = (%.4f, %.4f, %.4f)\n",
                   source.iso[0], source.iso[1], source.iso[2]);
            printf("\t axis   = (%.4f, %.4f, %.4f)\n",
                   source.beam_axis[0], source.beam_axis[1], source.beam_axis[2]);
        }
    }

    /* Slice 2f: open the phsp pair if `source type = phase_space`. The
     * rotated-beam basis above is required; the analytic sampler is
     * bypassed in initHistory when phsp_active is set. */
    if (phsp_requested) {
        if (!source.rotated_beam) {
            printf("phase_space mode requires a rotated-beam basis "
                   "(source position / isocenter / collimator right / "
                   "collimator up).\n");
            exit(EXIT_FAILURE);
        }
        char phsp_header[BUFFER_SIZE];
        if (getInputValue(phsp_header, "phsp file") != 1) {
            printf("phase_space mode: missing 'phsp file' key.\n");
            exit(EXIT_FAILURE);
        }
        FILE *fh = fopen(phsp_header, "r");
        if (fh == NULL) {
            printf("Unable to open phsp header file: %s\n", phsp_header);
            exit(EXIT_FAILURE);
        }
        long long total_records = -1;
        int record_length = -1;
        char section[64] = "";
        char hline[1024];
        while (fgets(hline, sizeof(hline), fh) != NULL) {
            char *t = hline;
            while (*t == ' ' || *t == '\t') t++;
            char *end = t + strlen(t);
            while (end > t && (end[-1] == '\n' || end[-1] == '\r' ||
                               end[-1] == ' ' || end[-1] == '\t')) end--;
            *end = '\0';
            if (*t == '\0' || *t == '#' || *t == '!') continue;
            if (*t == '$') {
                char *colon = strchr(t, ':');
                if (colon) *colon = '\0';
                snprintf(section, sizeof(section), "%s", t + 1);
                continue;
            }
            if (strcmp(section, "RECORD_LENGTH") == 0) {
                record_length = atoi(t);
            } else if (strcmp(section, "PARTICLES") == 0) {
                total_records = atoll(t);
            }
        }
        fclose(fh);
        if (record_length != 32 || total_records <= 0) {
            printf("phsp header %s: only $RECORD_LENGTH=32 with positive "
                   "$PARTICLES is supported.\n", phsp_header);
            exit(EXIT_FAILURE);
        }
        /* Derive .IAEAphsp path. */
        char phsp_data[BUFFER_SIZE];
        size_t hlen = strlen(phsp_header);
        const char *hsuffix = ".IAEAheader";
        size_t slen = strlen(hsuffix);
        if (hlen < slen ||
            strcmp(phsp_header + hlen - slen, hsuffix) != 0) {
            printf("phsp header path must end with '.IAEAheader': %s\n",
                   phsp_header);
            exit(EXIT_FAILURE);
        }
        memcpy(phsp_data, phsp_header, hlen - slen);
        memcpy(phsp_data + (hlen - slen), ".IAEAphsp", strlen(".IAEAphsp") + 1);
        source.phsp_fp = fopen(phsp_data, "rb");
        if (source.phsp_fp == NULL) {
            printf("Unable to open phsp data file: %s\n", phsp_data);
            exit(EXIT_FAILURE);
        }
        source.phsp_active = 1;
        source.phsp_total_records = total_records;
        source.phsp_records_read = 0;
        if (verbose_flag) {
            printf("Phase-space mode enabled (%lld records).\n",
                   source.phsp_total_records);
        }
    }

    return;
}

void cleanSource() {

    free(source.cdfinv1);
    free(source.cdfinv2);
    if (source.phsp_fp != NULL) {
        fclose(source.phsp_fp);
        source.phsp_fp = NULL;
    }

    return;
}

/******************************************************************************/
/* Scoring definitions */
struct Score {
    double ensrc;               // total energy from source
    double *endep;              // 3D dep. energy matrix per batch
    
    /* The following variables are needed for statistical analysis. Their
     values are accumulated across the simulation */
    double *accum_endep;        // 3D deposited energy matrix
    double *accum_endep2;       // 3D square deposited energy
};
struct Score score;

void initScore() {
    
    int gridsize = geometry.isize*geometry.jsize*geometry.ksize;
    
    score.ensrc = 0.0;
    
    /* Region with index 0 corresponds to region outside phantom */
    score.endep = malloc((gridsize + 1)*sizeof(double));
    score.accum_endep = malloc((gridsize + 1)*sizeof(double));
    score.accum_endep2 = malloc((gridsize + 1)*sizeof(double));
    
    /* Initialize all arrays to zero */
    memset(score.endep, 0.0, (gridsize + 1)*sizeof(double));
    memset(score.accum_endep, 0.0, (gridsize + 1)*sizeof(double));
    memset(score.accum_endep2, 0.0, (gridsize + 1)*sizeof(double));
    
    return;
}

void cleanScore() {
    
    free(score.endep);
    free(score.accum_endep);
    free(score.accum_endep2);
    
    return;
}

void ausgab(double edep) {
    
    int np = stack.np;
    int irl = stack.ir[np];
    double endep = stack.wt[np]*edep;
        
    /* Deposit particle energy on spot */
    #pragma omp atomic
    score.endep[irl] += endep;
    
    return;
}

void accumEndep() {
    
    int gridsize = geometry.isize*geometry.jsize*geometry.ksize;
    
    /* Accumulate endep and endep squared for statistical analysis */
    double edep = 0.0;
    
    int irl = 0;
    
    #pragma omp parallel for firstprivate(edep)
    for (irl=0; irl<gridsize + 1; irl++) {
        edep = score.endep[irl];
        
        score.accum_endep[irl] += edep;
        score.accum_endep2[irl] += edep*edep;
    }
    
    /* Clean scoring array */
    memset(score.endep, 0.0, (gridsize + 1)*sizeof(double));
    
    return;
}

void accumulateResults(int iout, int nhist, int nbatch)
{
    int irl;
    int imax = geometry.isize;
    int ijmax = geometry.isize*geometry.jsize;
    double endep, endep2, unc_endep;

    /* Calculate incident fluence */
    double inc_fluence = (double)nhist;
    double mass;
    int iz;

    #pragma omp parallel for private(irl,endep,endep2,unc_endep,mass)
    for (iz=0; iz<geometry.ksize; iz++) {
        for (int iy=0; iy<geometry.jsize; iy++) {
            for (int ix=0; ix<geometry.isize; ix++) {
                irl = 1 + ix + iy*imax + iz*ijmax;
                endep = score.accum_endep[irl];
                endep2 = score.accum_endep2[irl];
                
                /* First calculate mean deposited energy across batches and its
                 uncertainty */
                endep /= (double)nbatch;
                endep2 /= (double)nbatch;
                
                /* Batch approach uncertainty calculation */
                if (endep != 0.0) {
                    unc_endep = endep2 - endep*endep;
                    unc_endep /= (double)(nbatch - 1);
                    
                    /* Relative uncertainty */
                    unc_endep = sqrt(unc_endep)/endep;
                }
                else {
                    endep = 0.0;
                    unc_endep = 0.9999999;
                }
                
                /* We separate de calculation of dose, to give the user the
                 option to output mean energy (iout=0) or deposited dose
                 (iout=1) per incident fluence */
                
                if (iout) {
                    
                    /* Convert deposited energy to dose */
                    mass = (geometry.xbounds[ix+1] - geometry.xbounds[ix])*
                        (geometry.ybounds[iy+1] - geometry.ybounds[iy])*
                        (geometry.zbounds[iz+1] - geometry.zbounds[iz]);
                    
                    /* Transform deposited energy to Gy */
                    mass *= geometry.med_densities[irl-1];
                    endep *= 1.602E-10/(mass*inc_fluence);
                    
                } else {    /* Output mean deposited energy */
                    endep /= inc_fluence;
                }
                
                /* Store output quantities */
                score.accum_endep[irl] = endep;
                score.accum_endep2[irl] = unc_endep;
            }
        }
    }
    
    /* Zero dose in air */
    #pragma omp parallel for private(irl)
    for (iz=0; iz<geometry.ksize; iz++) {
        for (int iy=0; iy<geometry.jsize; iy++) {
            for (int ix=0; ix<geometry.isize; ix++) {
                irl = 1 + ix + iy*imax + iz*ijmax;
                
                if(geometry.med_densities[irl-1] < 0.044) {
                    score.accum_endep[irl] = 0.0;
                    score.accum_endep2[irl] = 0.9999999;
                }
            }
        }
    }
    
    return;
}

void outputResults(char *output_file, int iout, int nhist, int nbatch) {
    
    // Accumulate the results
    accumulateResults(iout, nhist,nbatch);
    
    int irl;
    int imax = geometry.isize;
    int ijmax = geometry.isize*geometry.jsize;
    
    /* Output to file */
    char extension[15];
    if (iout) {
        strcpy(extension, ".3ddose");
    } else {
        strcpy(extension, ".3denergy");
    }
    
    /* Get file path from input data */
    char output_folder[128];
    char buffer[BUFFER_SIZE];
    
    if (getInputValue(buffer, "output folder") != 1) {
        printf("Can not find 'output folder' key on input file.\n");
        exit(EXIT_FAILURE);
    }
    removeSpaces(output_folder, buffer);
    
    /* Make space for the new string */
    char* file_name = malloc(strlen(output_folder) + strlen(output_file) + 
        strlen(extension) + 1);
    strcpy(file_name, output_folder);
    strcat(file_name, output_file); /* add the file name */
    strcat(file_name, extension); /* add the extension */
    
    FILE *fp;
    if ((fp = fopen(file_name, "w")) == NULL) {
        printf("Unable to open file: %s\n", file_name);
        exit(EXIT_FAILURE);
    }
    
    /* Grid dimensions */
    fprintf(fp, "%5d%5d%5d\n",
            geometry.isize, geometry.jsize, geometry.ksize);
    
    /* Boundaries in x-, y- and z-directions */
    for (int ix = 0; ix<=geometry.isize; ix++) {
        fprintf(fp, "%f ", geometry.xbounds[ix]);
    }
    fprintf(fp, "\n");
    for (int iy = 0; iy<=geometry.jsize; iy++) {
        fprintf(fp, "%f ", geometry.ybounds[iy]);
    }
    fprintf(fp, "\n");
    for (int iz = 0; iz<=geometry.ksize; iz++) {
        fprintf(fp, "%f ", geometry.zbounds[iz]);
    }
    fprintf(fp, "\n");
    
    /* Dose or energy array */
    for (int iz=0; iz<geometry.ksize; iz++) {
        for (int iy=0; iy<geometry.jsize; iy++) {
            for (int ix=0; ix<geometry.isize; ix++) {
                irl = 1 + ix + iy*imax + iz*ijmax;
                fprintf(fp, "%e ", score.accum_endep[irl]);
            }
        }
    }
    fprintf(fp, "\n");
    
    /* Uncertainty array */
    for (int iz=0; iz<geometry.ksize; iz++) {
        for (int iy=0; iy<geometry.jsize; iy++) {
            for (int ix=0; ix<geometry.isize; ix++) {
                irl = 1 + ix + iy*imax + iz*ijmax;
                fprintf(fp, "%f ", score.accum_endep2[irl]);
            }
        }
    }
    fprintf(fp, "\n");
    
    /* Cleaning */
    fclose(fp);
    free(file_name);

    return;
}

/******************************************************************************/
/* Region-by-region definitions */
void initRegions() {
    
    /* +1 : consider region surrounding phantom */
    int nreg = geometry.isize*geometry.jsize*geometry.ksize + 1;
    
    /* Allocate memory for region data */
    region.med = malloc(nreg*sizeof(int));
    region.rhof = malloc(nreg*sizeof(double));
    region.pcut = malloc(nreg*sizeof(double));
    region.ecut = malloc(nreg*sizeof(double));
    
    /* First get global energy cutoff parameters */
    char buffer[BUFFER_SIZE];
    if (getInputValue(buffer, "global ecut") != 1) {
        printf("Can not find 'global ecut' key on input file.\n");
        exit(EXIT_FAILURE);
    }
    double ecut = atof(buffer);
    
    if (getInputValue(buffer, "global pcut") != 1) {
        printf("Can not find 'global pcut' key on input file.\n");
        exit(EXIT_FAILURE);
    }
    double pcut = atof(buffer);
    
    /* Initialize transport parameters on each region. Region 0 is outside the
     geometry */
    region.med[0] = VACUUM;
    region.rhof[0] = 0.0;
    region.pcut[0] = 0.0;
    region.ecut[0] = 0.0;
    
    for (int i=1; i<nreg; i++) {
        
        /* -1 : EGS counts media from 1. Substract 1 to get medium index */
        int imed = geometry.med_indices[i - 1] - 1;
        region.med[i] = imed;
        
        if (imed == VACUUM) {
            region.rhof[0] = 0.0F;
            region.pcut[0] = 0.0F;
            region.ecut[0] = 0.0F;
        }
        else {
            if (geometry.med_densities[i - 1] == 0.0F) {
                region.rhof[i] = 1.0;
            }
            else {
                region.rhof[i] =
                    geometry.med_densities[i - 1]/pegs_data.rho[imed];
            }
            
            /* Check if global cut-off values are within PEGS data */
            if (pegs_data.ap[imed] <= pcut) {
                region.pcut[i] = pcut;
            } else {
                printf("Warning!, global pcut value is below PEGS's pcut value "
                       "%f for medium %d, using PEGS value.\n",
                       pegs_data.ap[imed], imed);
                region.pcut[i] = pegs_data.ap[imed];
            }
            if (pegs_data.ae[imed] <= ecut) {
                region.ecut[i] = ecut;
            } else {
                printf("Warning!, global pcut value is below PEGS's ecut value "
                       "%f for medium %d, using PEGS value.\n",
                       pegs_data.ae[imed], imed);
            }
        }
    }
    
    return;
}

void initHistory() {

    double rnno1;
    double rnno2;
    
    /* Initialize first particle of the stack from source data */
    stack.np = 0;
    stack.iq[stack.np] = source.charge;
    
    /* Get primary particle energy */
    double ein = 0.0;
    if (source.spectrum) {
        /* Sample initial energy from spectrum data */
        rnno1 = setRandom();
        rnno2 = setRandom();
        
        /* Sample bin number in order to select particle energy */
        int k = (int)fmin(source.deltak*rnno1, source.deltak - 1.0);
        ein = source.cdfinv1[k] + rnno2*source.cdfinv2[k];
    }
    else {
        /* Monoenergetic source */
        ein = source.energy;
    }
    
    /* Check if the particle is an electron, in such a case add electron
     rest mass energy */
    if (stack.iq[stack.np] != 0) {
        /* Electron or positron */
        stack.e[stack.np] = ein + RM;
    }
    else {
        /* Photon */
        stack.e[stack.np] = ein;
    }
    
    /* Accumulate sampled kinetic energy for fraction of deposited energy
     calculations */
    score.ensrc += ein;
           
    /* Slice 2f: phase-space mode. Read one phsp record, transform from
     * the iso-plane basis into the MVompMC voxel-grid frame, then run
     * the same ray-box-intersect-plus-region-index logic as rotated-
     * beam mode. */
    if (source.phsp_active) {
        if (source.phsp_records_read >= source.phsp_total_records) {
            /* Phsp exhausted before histories ran out. Rewind so the
             * file can be re-used for the remaining histories. */
            fseek(source.phsp_fp, 0L, SEEK_SET);
            source.phsp_records_read = 0;
        }
        unsigned char rbuf[32];
        if (fread(rbuf, 1, sizeof(rbuf), source.phsp_fp) != sizeof(rbuf)) {
            printf("Short read on phsp data file at record %lld.\n",
                   source.phsp_records_read);
            exit(EXIT_FAILURE);
        }
        source.phsp_records_read++;

        unsigned char type_byte = rbuf[0];
        int phsp_type = type_byte & 0x7F;
        int w_negative = (type_byte & 0x80) ? 1 : 0;
        float fE, fx, fy, fu, fv, fweight;
        memcpy(&fE,      rbuf + 4,  4);
        memcpy(&fx,      rbuf + 8,  4);
        memcpy(&fy,      rbuf + 12, 4);
        memcpy(&fu,      rbuf + 16, 4);
        memcpy(&fv,      rbuf + 20, 4);
        memcpy(&fweight, rbuf + 24, 4);

        /* Override charge / energy from phsp; the spectrum sampling
         * earlier in initHistory is discarded for this particle. */
        switch (phsp_type) {
            case 1: stack.iq[stack.np] = 0;  break;  /* photon */
            case 2: stack.iq[stack.np] = -1; break;  /* electron */
            case 3: stack.iq[stack.np] = 1;  break;  /* positron */
            default:
                printf("Unsupported phsp particle type %d at record %lld.\n",
                       phsp_type, source.phsp_records_read - 1);
                exit(EXIT_FAILURE);
        }
        if (stack.iq[stack.np] != 0) {
            stack.e[stack.np] = (double)fE + RM;
        } else {
            stack.e[stack.np] = (double)fE;
        }

        double w_phsp = sqrt(fmax(0.0, 1.0 - (double)fu*fu - (double)fv*fv));
        if (w_negative) w_phsp = -w_phsp;
        double px = fx, py = fy;
        double tx = source.iso[0] + px*source.cright[0] + py*source.cup[0];
        double ty = source.iso[1] + px*source.cright[1] + py*source.cup[1];
        double tz = source.iso[2] + px*source.cright[2] + py*source.cup[2];
        double ux = (double)fu*source.cright[0] + (double)fv*source.cup[0]
                  + w_phsp*source.beam_axis[0];
        double uy = (double)fu*source.cright[1] + (double)fv*source.cup[1]
                  + w_phsp*source.beam_axis[1];
        double uz = (double)fu*source.cright[2] + (double)fv*source.cup[2]
                  + w_phsp*source.beam_axis[2];

        /* Ray-box intersect from (tx,ty,tz) along (ux,uy,uz). For phsp
         * particles, the iso-plane sample point may already be inside
         * the phantom AABB; tmin clamps to 0 in that case. */
        double xmin = geometry.xbounds[0];
        double xmax = geometry.xbounds[geometry.isize];
        double ymin = geometry.ybounds[0];
        double ymax = geometry.ybounds[geometry.jsize];
        double zmin = geometry.zbounds[0];
        double zmax = geometry.zbounds[geometry.ksize];
        double tmin = -1e300, tmax = 1e300;
        for (int axis = 0; axis < 3; axis++) {
            double o = (axis == 0) ? tx : (axis == 1) ? ty : tz;
            double d = (axis == 0) ? ux : (axis == 1) ? uy : uz;
            double lo = (axis == 0) ? xmin : (axis == 1) ? ymin : zmin;
            double hi = (axis == 0) ? xmax : (axis == 1) ? ymax : zmax;
            if (fabs(d) < 1e-12) {
                if (o < lo || o > hi) { tmax = -1.0; break; }
                continue;
            }
            double t1 = (lo - o) / d;
            double t2 = (hi - o) / d;
            if (t1 > t2) { double s = t1; t1 = t2; t2 = s; }
            if (t1 > tmin) tmin = t1;
            if (t2 < tmax) tmax = t2;
            if (tmax < tmin) break;
        }
        if (tmax < 0.0 || tmax < tmin) {
            /* Particle's ray misses the phantom — discard. */
            stack.x[stack.np] = tx; stack.y[stack.np] = ty; stack.z[stack.np] = tz;
            stack.u[stack.np] = ux; stack.v[stack.np] = uy; stack.w[stack.np] = uz;
            stack.ir[stack.np] = 0;
            stack.wt[stack.np] = 0.0;
            stack.dnear[stack.np] = 0.0;
            return;
        }
        double t_entry = tmin > 0.0 ? tmin : 0.0;
        double ex = tx + ux * t_entry;
        double ey = ty + uy * t_entry;
        double ez = tz + uz * t_entry;
        const double NUDGE = 1e-9;
        if (ex < xmin) ex = xmin + NUDGE; else if (ex > xmax) ex = xmax - NUDGE;
        if (ey < ymin) ey = ymin + NUDGE; else if (ey > ymax) ey = ymax - NUDGE;
        if (ez < zmin) ez = zmin + NUDGE; else if (ez > zmax) ez = zmax - NUDGE;

        stack.x[stack.np] = ex;
        stack.y[stack.np] = ey;
        stack.z[stack.np] = ez;
        stack.u[stack.np] = ux;
        stack.v[stack.np] = uy;
        stack.w[stack.np] = uz;
        int ix = 0, iy = 0, iz = 0;
        while (ix < geometry.isize - 1 && geometry.xbounds[ix+1] <= ex) ix++;
        while (iy < geometry.jsize - 1 && geometry.ybounds[iy+1] <= ey) iy++;
        while (iz < geometry.ksize - 1 && geometry.zbounds[iz+1] <= ez) iz++;
        stack.ir[stack.np] = 1 + ix
                               + iy * geometry.isize
                               + iz * geometry.isize * geometry.jsize;
        stack.wt[stack.np] = (double)fweight;
        stack.dnear[stack.np] = 0.0;
        return;
    }

    /* Rotated-beam mode (slice 2e). Sample on the iso plane, ray-box
     * intersect into the phantom AABB, set entry voxel's region. */
    if (source.rotated_beam) {
        double px = 0.0, py = 0.0;
        double tx = 0.0, ty = 0.0, tz = 0.0;
        double ux = 0.0, uy = 0.0, uz = 0.0;
        double rxyz_iso = 0.0, fw = 0.0, rnno3 = 0.0;
        int accepted = 0;
        /* Try at most a few times in case the collimator is set up so
         * that some samples miss the phantom AABB. In normal clinical
         * setups every sample hits, so this loop is dominated by the
         * cosine-cubed acceptance. */
        for (int attempt = 0; attempt < 100 && !accepted; attempt++) {
            rnno3 = setRandom();
            px = rnno3 * (source.xinu - source.xinl) + source.xinl;
            rnno3 = setRandom();
            py = rnno3 * (source.yinu - source.yinl) + source.yinl;
            tx = source.iso[0] + px*source.cright[0] + py*source.cup[0];
            ty = source.iso[1] + px*source.cright[1] + py*source.cup[1];
            tz = source.iso[2] + px*source.cright[2] + py*source.cup[2];
            double dx = tx - source.source_pos[0];
            double dy = ty - source.source_pos[1];
            double dz = tz - source.source_pos[2];
            rxyz_iso = sqrt(dx*dx + dy*dy + dz*dz);
            if (rxyz_iso == 0.0) continue;
            ux = dx / rxyz_iso;
            uy = dy / rxyz_iso;
            uz = dz / rxyz_iso;
            double cos_t = ux*source.beam_axis[0]
                         + uy*source.beam_axis[1]
                         + uz*source.beam_axis[2];
            fw = cos_t * cos_t * cos_t;
            rnno3 = setRandom();
            if (rnno3 >= fw) continue;

            /* Ray-box intersect with phantom AABB (slab method). */
            double sx = source.source_pos[0];
            double sy = source.source_pos[1];
            double sz = source.source_pos[2];
            double xmin = geometry.xbounds[0];
            double xmax = geometry.xbounds[geometry.isize];
            double ymin = geometry.ybounds[0];
            double ymax = geometry.ybounds[geometry.jsize];
            double zmin = geometry.zbounds[0];
            double zmax = geometry.zbounds[geometry.ksize];
            double tmin = -1e300, tmax = 1e300;
            for (int axis = 0; axis < 3; axis++) {
                double o = (axis == 0) ? sx : (axis == 1) ? sy : sz;
                double d = (axis == 0) ? ux : (axis == 1) ? uy : uz;
                double lo = (axis == 0) ? xmin : (axis == 1) ? ymin : zmin;
                double hi = (axis == 0) ? xmax : (axis == 1) ? ymax : zmax;
                if (fabs(d) < 1e-12) {
                    if (o < lo || o > hi) { tmax = -1.0; break; }
                    continue;
                }
                double t1 = (lo - o) / d;
                double t2 = (hi - o) / d;
                if (t1 > t2) { double s = t1; t1 = t2; t2 = s; }
                if (t1 > tmin) tmin = t1;
                if (t2 < tmax) tmax = t2;
                if (tmax < tmin) break;
            }
            if (tmax < 0.0 || tmax < tmin) {
                /* Ray misses the phantom from the source position. Try
                 * another sample. */
                continue;
            }
            double t_entry = tmin > 0.0 ? tmin : 0.0;
            double ex = sx + ux * t_entry;
            double ey = sy + uy * t_entry;
            double ez = sz + uz * t_entry;
            /* Nudge into the box if floating-point landed exactly on a
             * face. The voxel-finding loops below treat boundaries as
             * left-inclusive. */
            const double NUDGE = 1e-9;
            if (ex < xmin) ex = xmin + NUDGE; else if (ex > xmax) ex = xmax - NUDGE;
            if (ey < ymin) ey = ymin + NUDGE; else if (ey > ymax) ey = ymax - NUDGE;
            if (ez < zmin) ez = zmin + NUDGE; else if (ez > zmax) ez = zmax - NUDGE;

            stack.x[stack.np] = ex;
            stack.y[stack.np] = ey;
            stack.z[stack.np] = ez;
            stack.u[stack.np] = ux;
            stack.v[stack.np] = uy;
            stack.w[stack.np] = uz;

            int ix = 0, iy = 0, iz = 0;
            while (ix < geometry.isize - 1 && geometry.xbounds[ix+1] <= ex) ix++;
            while (iy < geometry.jsize - 1 && geometry.ybounds[iy+1] <= ey) iy++;
            while (iz < geometry.ksize - 1 && geometry.zbounds[iz+1] <= ez) iz++;
            stack.ir[stack.np] = 1 + ix
                                   + iy * geometry.isize
                                   + iz * geometry.isize * geometry.jsize;
            stack.wt[stack.np] = 1.0;
            stack.dnear[stack.np] = 0.0;
            accepted = 1;
        }
        if (!accepted) {
            /* Failed to hit the phantom after the rejection budget; emit
             * a zero-weight particle that will deposit nothing but keeps
             * the history-counter advancing. */
            stack.x[stack.np] = source.source_pos[0];
            stack.y[stack.np] = source.source_pos[1];
            stack.z[stack.np] = source.source_pos[2];
            stack.u[stack.np] = source.beam_axis[0];
            stack.v[stack.np] = source.beam_axis[1];
            stack.w[stack.np] = source.beam_axis[2];
            stack.ir[stack.np] = 0;
            stack.wt[stack.np] = 0.0;
            stack.dnear[stack.np] = 0.0;
        }
        return;
    }

    /* Set particle position. First obtain a random position in the rectangle
     defined by the collimator */
    double rxyz = 0.0;
    if (source.xsize == 0.0 || source.ysize == 0.0) {
        stack.x[stack.np] = source.xinl;
        stack.y[stack.np] = source.yinl;
        
        rxyz = sqrt(pow(source.ssd, 2.0) + pow(stack.x[stack.np], 2.0) +
                    pow(stack.y[stack.np], 2.0));
        
        /* Get direction along z-axis */
        stack.w[stack.np] = source.ssd/rxyz;
        
    } else {
        double fw;
        double rnno3;
        do { /* rejection sampling of the initial position */
            rnno3 = setRandom();
            stack.x[stack.np] = rnno3*source.xsize + source.xinl;
            rnno3 = setRandom();
            stack.y[stack.np] = rnno3*source.ysize + source.yinl;
            rnno3 = setRandom();
            rxyz = sqrt(source.ssd*source.ssd + 
				stack.x[stack.np]*stack.x[stack.np] +
				stack.y[stack.np]*stack.y[stack.np]);
            
            /* Get direction along z-axis */
            stack.w[stack.np] = source.ssd/rxyz;
            fw = stack.w[stack.np]*stack.w[stack.np]*stack.w[stack.np];
        } while(rnno3 >= fw);
    }
    /* Set position of the particle in front of the geometry */
    stack.z[stack.np] = geometry.zbounds[0];
    
    /* At this point the position has been found, calculate particle
     direction */
    stack.u[stack.np] = stack.x[stack.np]/rxyz;
    stack.v[stack.np] = stack.y[stack.np]/rxyz;
    
    /* Determine region index of source particle */
    int ix, iy;
    if (source.xsize == 0.0) {
        ix = source.ixinl;
    } else {
        ix = source.ixinl - 1;
        while ((geometry.xbounds[ix+1] < stack.x[stack.np]) && ix < geometry.isize-1) {
            ix++;
        }
    }
    if (source.ysize == 0.0) {
        iy = source.iyinl;
    } else {
        iy = source.iyinl - 1;
        while ((geometry.ybounds[iy+1] < stack.y[stack.np]) && iy < geometry.jsize-1) {
            iy++;
        }
    }
    stack.ir[stack.np] = 1 + ix + iy*geometry.isize;
    
    /* Set statistical weight and distance to closest boundary*/
    stack.wt[stack.np] = 1.0;
    stack.dnear[stack.np] = 0.0;
        
    return;
}

/******************************************************************************/
/* omc_dosxyz_dump_dose: serialise the post-simulation dose grid to a
 * file descriptor in a small, fixed binary format.
 *
 * Intended to be called once, after omc_dosxyz_main returns 0 in the
 * caller's child process — at that point `geometry` and `score` are
 * fully populated and `outputResults()` has accumulated the per-voxel
 * dose and relative uncertainty into `score.accum_endep` /
 * `score.accum_endep2`. The function reads only existing globals; it
 * does not allocate or modify simulation state.
 *
 * Format (little-endian; values are doubles unless noted):
 *
 *   char     magic[4]   = "QDD1"
 *   uint32_t version    = 1
 *   int32_t  nx, ny, nz
 *   double   spacing_cm[3]   (xbounds[1]-xbounds[0], y..., z...)
 *   double   origin_cm[3]    (xbounds[0], ybounds[0], zbounds[0])
 *   double   dose[nx*ny*nz]  (X fastest, Z slowest — same as Dose3d)
 *   double   unc[nx*ny*nz]   (relative uncertainty in [0..1])
 *
 * Returns 0 on success, -1 if any write() returns short. The caller
 * is responsible for closing fd. */
int omc_dosxyz_dump_dose(int fd) {
    int isize = geometry.isize;
    int jsize = geometry.jsize;
    int ksize = geometry.ksize;
    int64_t n = (int64_t)isize * jsize * ksize;
    int imax = isize;
    int ijmax = isize * jsize;

    /* Header. */
    {
        unsigned char hdr[4 + 4 + 12 + 24 + 24];
        size_t off = 0;
        memcpy(hdr + off, "QDD1", 4); off += 4;
        uint32_t v = 1;
        memcpy(hdr + off, &v, 4); off += 4;
        int32_t dims[3] = {(int32_t)isize, (int32_t)jsize, (int32_t)ksize};
        memcpy(hdr + off, dims, sizeof(dims)); off += sizeof(dims);
        double spacing[3] = {
            geometry.xbounds[1] - geometry.xbounds[0],
            geometry.ybounds[1] - geometry.ybounds[0],
            geometry.zbounds[1] - geometry.zbounds[0],
        };
        memcpy(hdr + off, spacing, sizeof(spacing)); off += sizeof(spacing);
        double origin[3] = {
            geometry.xbounds[0],
            geometry.ybounds[0],
            geometry.zbounds[0],
        };
        memcpy(hdr + off, origin, sizeof(origin)); off += sizeof(origin);
        ssize_t wrote = write(fd, hdr, off);
        if (wrote < 0 || (size_t)wrote != off) return -1;
    }

    /* Dose values, in the same loop order as outputResults() (X
     * fastest, Z slowest), so the receiver sees Dose3d-compatible
     * ordering with no permutation. The kernel's `score` array is
     * 1-indexed (irl = 1 + ix + iy*imax + iz*ijmax); we walk it
     * voxel-by-voxel and stream a small chunk at a time so the
     * function works against pipes with bounded buffers. */
    enum { CHUNK = 4096 };
    double buf[CHUNK];
    int64_t streamed = 0;

    /* dose[] */
    streamed = 0;
    while (streamed < n) {
        int64_t batch = (n - streamed) < CHUNK ? (n - streamed) : CHUNK;
        for (int64_t k = 0; k < batch; k++) {
            int64_t lin = streamed + k;
            int iz = (int)(lin / ijmax);
            int rem = (int)(lin - (int64_t)iz * ijmax);
            int iy = rem / imax;
            int ix = rem - iy * imax;
            int irl = 1 + ix + iy * imax + iz * ijmax;
            buf[k] = score.accum_endep[irl];
        }
        ssize_t want = (ssize_t)(batch * sizeof(double));
        ssize_t got = write(fd, buf, (size_t)want);
        if (got < 0 || got != want) return -1;
        streamed += batch;
    }

    /* uncertainty[] */
    streamed = 0;
    while (streamed < n) {
        int64_t batch = (n - streamed) < CHUNK ? (n - streamed) : CHUNK;
        for (int64_t k = 0; k < batch; k++) {
            int64_t lin = streamed + k;
            int iz = (int)(lin / ijmax);
            int rem = (int)(lin - (int64_t)iz * ijmax);
            int iy = rem / imax;
            int ix = rem - iy * imax;
            int irl = 1 + ix + iy * imax + iz * ijmax;
            buf[k] = score.accum_endep2[irl];
        }
        ssize_t want = (ssize_t)(batch * sizeof(double));
        ssize_t got = write(fd, buf, (size_t)want);
        if (got < 0 || got != want) return -1;
        streamed += batch;
    }
    return 0;
}

/******************************************************************************/
/* omc_dosxyz main function.
 *
 * When compiled with -DOMC_DOSXYZ_AS_LIBRARY (set by clinical-dosecalc's
 * CMake when linking this translation unit into qdc_core), the symbol
 * `main` is renamed to `omc_dosxyz_main` via the macro below so the
 * function can coexist with a hosting program's own main(). The
 * standalone executable build leaves this token alone, so the binary
 * still has a real main() and ./omc_dosxyz keeps working unchanged.
 *
 * The header omc_dosxyz.h declares the renamed symbol so callers can
 * invoke the kernel without going through fork+exec. */

/* Cooperative-cancellation flag (slice 5c). Set via SIGINT handler
 * installed below; the batch loop checks it at each boundary and
 * stops cleanly. Hosting processes (e.g. qdc) may also set it
 * directly to drive cancellation programmatically. */
volatile sig_atomic_t omc_dosxyz_cancel = 0;

static void omc_dosxyz_handle_sigint(int sig) {
    (void)sig;
    omc_dosxyz_cancel = 1;
}

#ifdef OMC_DOSXYZ_AS_LIBRARY
#define main omc_dosxyz_main
#endif
int main (int argc, char **argv) {

    /* Reset cancellation state for this run. The flag is process-
     * global, but for any single invocation we want a clean slate. */
    omc_dosxyz_cancel = 0;
    /* Install SIGINT handler so Ctrl+C in a terminal flips the flag
     * instead of killing the process mid-simulation. The handler is
     * idempotent — setting it twice (e.g. when the host already had
     * one) just installs ours. The default-action handler is
     * inherited from whatever the caller had before this; we don't
     * try to restore it on exit because main() is the kernel's
     * entire lifecycle for this process. */
    signal(SIGINT, omc_dosxyz_handle_sigint);

    /* Execution time measurement */
    double tbegin;
    tbegin = omc_get_time();
    
    /* Parsing program options */
    
    int c;
    char *input_file = NULL;
    char *output_file = NULL;
    /* When non-negative, dump the post-simulation dose grid to this fd
     * in the QDD1 binary format (see omc_dosxyz_dump_dose). Set via
     * --dump-dose-fd N. Used by qdc's in-process integration to
     * receive the dose without parsing the .3ddose text file. */
    int dump_dose_fd = -1;

    while (1) {
        static struct option long_options[] =
        {
            /* These options set a flag. */
            {"verbose", no_argument, &verbose_flag, 1},
            {"brief",   no_argument, &verbose_flag, 0},
            /* These options don’t set a flag.
             We distinguish them by their indices. */
            {"input",  required_argument, 0, 'i'},
            {"output",    required_argument, 0, 'o'},
            {"dump-dose-fd", required_argument, 0, 'D'},
            {0, 0, 0, 0}
        };

        /* getopt_long stores the option index here. */
        int option_index = 0;

        c = getopt_long(argc, argv, "i:o:D:",
                         long_options, &option_index);

        /* Detect the end of the options. */
        if (c == -1)
            break;

        switch (c) {
            case 0:
                /* If this option set a flag, do nothing else now. */
                if (long_options[option_index].flag != 0)
                    break;
                printf ("option %s", long_options[option_index].name);
                if (optarg)
                    printf (" with arg %s", optarg);
                printf ("\n");
                break;

            case 'i':
                input_file = malloc(strlen(optarg) + 1);
                strcpy(input_file, optarg);
                printf ("option -i with value `%s'\n", input_file);
                break;

            case 'o':
                output_file = malloc(strlen(optarg) + 1);
                strcpy(output_file, optarg);
                printf ("option -o with value `%s'\n", output_file);
                break;

            case 'D':
                dump_dose_fd = atoi(optarg);
                break;

            case '?':
                /* getopt_long already printed an error message. */
                break;

            default:
                exit(EXIT_FAILURE);
        }
    }
    
    /* Instead of reporting ‘--verbose’
     and ‘--brief’ as they are encountered,
     we report the final status resulting from them. */
    if (verbose_flag)
        puts ("verbose flag is set");
    
    /* Print any remaining command line arguments (not options). */
    if (optind < argc)
    {
        printf ("non-option ARGV-elements: ");
        while (optind < argc)
            printf ("%s ", argv[optind++]);
        putchar ('\n');
    }
    
    /* Parse input file and print key,value pairs (test) */
    parseInputFile(input_file);

    /* Get information of OpenMP environment. Honour OMP_NUM_THREADS
     * if it is set in the environment; otherwise default to all
     * available processors. This lets external callers (e.g. byte-
     * identity regression tests) force serial execution without
     * recompiling. */
#ifdef _OPENMP
    int omp_size = omp_get_num_procs();
    const char *omp_env = getenv("OMP_NUM_THREADS");
    if (omp_env != NULL && *omp_env != '\0') {
        int requested = atoi(omp_env);
        if (requested > 0) {
            omp_size = requested;
        }
    }
    printf("Number of OpenMP threads: %d\n", omp_size);
    omp_set_num_threads(omp_size);
#else
    printf("ompMC compiled without OpenMP support. Serial execution.\n");
#endif
    
    /* Read geometry information from phantom file and initialize geometry */
    initPhantom();
    
    /* With number of media and media names initialize the medium data */
    initMediaData();
    
    /* Initialize radiation source */
    initSource();
    
    /* Initialize data on a region-by-region basis */
    initRegions();

    /* Initialize VRT data */
    initVrt();
    
    /* Preparation of scoring struct */
    initScore();

    #pragma omp parallel
    {
      /* Initialize random number generator */
      initRandom();

      /* Initialize particle stack */
      initStack();
    }


    /* In verbose mode, list interaction data to output folder */
    if (verbose_flag) {
        listRayleigh();
        listPair();
        listPhoton();
        listElectron();
        listMscat();
        listSpin();
    }
    
    /* Shower call */
    
    /* Get number of histories and statistical batches */
    char buffer[BUFFER_SIZE];
    if (getInputValue(buffer, "ncase") != 1) {
        printf("Can not find 'ncase' key on input file.\n");
        exit(EXIT_FAILURE);
    }
   int nhist = atoi(buffer);
    
    if (getInputValue(buffer, "nbatch") != 1) {
        printf("Can not find 'nbatch' key on input file.\n");
        exit(EXIT_FAILURE);
    }
    int nbatch = atoi(buffer);
    
    if (nhist/nbatch == 0) {
        nhist = nbatch;
    }
    
    int nperbatch = nhist/nbatch;
    nhist = nperbatch*nbatch;
    
    int gridsize = geometry.isize*geometry.jsize*geometry.ksize;
    
    printf("Total number of particle histories: %d\n", nhist);
    printf("Number of statistical batches: %d\n", nbatch);
    printf("Histories per batch: %d\n", nperbatch);
    
    /* Execution time up to this point */
    printf("Execution time up to this point : %8.2f seconds\n",
           (omc_get_time() - tbegin));
    
    int ibatch;
    for (ibatch = 0; ibatch < nbatch; ibatch++) {
        /* Cooperative cancellation point (slice 5c). The flag is set
         * by SIGINT (Ctrl+C) or by the host directly. Breaking here
         * — at the top of an unstarted batch — keeps every completed
         * batch fully accumulated, so the partial dose we write
         * below is statistically meaningful. */
        if (omc_dosxyz_cancel) {
            printf("\nCancellation requested; finalising at batch %d/%d.\n",
                   ibatch, nbatch);
            fflush(stdout);
            break;
        }
        if (ibatch == 0) {
            /* Print header for information during simulation */
            printf("%-10s\t%-15s\t%-10s\n", "Batch #", "Elapsed time",
                   "RNG state");
            printf("%-10d\t%-15.2f\t%-5d%-5d\n", ibatch,
                   (omc_get_time() - tbegin), rng.ixx, rng.jxx);
        }
        else {
            /* Print state of current batch */
            printf("%-10d\t%-15.2f\t%-5d%-5d\n", ibatch,
                   (omc_get_time() - tbegin), rng.ixx, rng.jxx);

        }
        /* Stream progress in real time so a host process (or the
         * user) sees per-batch output as it lands instead of buffered
         * up to end-of-simulation. */
        fflush(stdout);
        int ihist;
        #pragma omp parallel for schedule(dynamic)
        for (ihist=0; ihist<nperbatch; ihist++) {
            /* Initialize particle history */
            initHistory();

            /* Start electromagnetic shower simulation */
            shower();
        }

        /* Accumulate results of current batch for statistical analysis */
        accumEndep();
    }
    /* Number of batches whose histories actually finished. On a clean
     * run this equals nbatch; on a cancellation it's the count
     * iterated past the cancel check. We pass this to outputResults
     * so the per-voxel mean and variance normalise against the right
     * denominator. */
    int completed_batches = ibatch;

    /* Print some output and execution time up to this point */
    if (omc_dosxyz_cancel) {
        printf("Simulation cancelled\n");
    } else {
        printf("Simulation finished\n");
    }
    printf("Execution time up to this point : %8.2f seconds\n",
           (omc_get_time() - tbegin));

    /* Analysis and output of results */
    if (verbose_flag) {
        /* Sum energy deposition in the phantom */
        double etot = 0.0;
        for (int irl=1; irl<gridsize+1; irl++) {
            etot += score.accum_endep[irl];
        }
        printf("Fraction of incident energy deposited in the phantom: %5.4f\n",
               etot/score.ensrc);
    }

    int iout = 1;   /* i.e. deposit mean dose per particle fluence */
    if (completed_batches < 2) {
        /* The accumulator's variance estimate divides by
         * (nbatch - 1), so anything less than 2 completed batches
         * can't produce a usable .3ddose. Skip the writes — the
         * caller will see exit code 0 (cancellation is not an
         * error) and an empty/missing dose file. */
        fprintf(stderr,
                "omc_dosxyz: only %d batch(es) completed; skipping dose "
                "output (need at least 2 for variance estimate).\n",
                completed_batches);
    } else {
        /* nhist arg to outputResults is histories-per-batch (the
         * per-history fluence divisor); we pass it through unchanged.
         * Only nbatch shifts to completed_batches so the per-batch
         * average uses the right denominator on a partial run. */
        outputResults(output_file, iout, nperbatch, completed_batches);

        /* If a host process asked for the post-simulation dose grid via
         * --dump-dose-fd N, serialise it now BEFORE cleanScore() frees the
         * accumulators. */
        if (dump_dose_fd >= 0) {
            if (omc_dosxyz_dump_dose(dump_dose_fd) != 0) {
                fprintf(stderr, "omc_dosxyz: dose dump to fd %d failed.\n",
                        dump_dose_fd);
            }
        }
    }

    /* Cleaning */
    cleanPhantom();
    cleanPhoton();
    cleanRayleigh();
    cleanPair();
    cleanElectron();
    cleanMscat();
    cleanSpin();
    cleanRegions();
    cleanScore();
    cleanSource();
    #pragma omp parallel
    {
      cleanRandom();
      cleanStack();
    }
    free(input_file);
    free(output_file);
    /* Get total execution time */
    printf("Total execution time : %8.5f seconds\n",
           (omc_get_time() - tbegin));
    
    exit (EXIT_SUCCESS);
}
