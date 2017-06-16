
/*
    findvalue(x, y, z, param) - returns interpolated value.
    azimuthalbounds(azi) - finds the bounding radial slices.
    radialbounds(rad, apnt) - finds the bounding vertical slices.
    verticalbounds(alt, apnt, rpnt) - finds the bounding cells.
    linterpolate(x, x0, x1, y0, y1) - interpolates values.

    TODO: Include a better treatment of the upper regions of the disk. Or just
    hope that they're sufficiently tenuous that they make little difference...
*/

double linterpolate(double x, double xa, double xb, double ya, double yb){
    return (x - xa) * (yb - ya) / (xb - xa) + ya;
}


int azimuthalbounds(double azi){
    int i;
    for (i=1; i<(NCELLS-1); i++) {
        if ((c3arr[i] - azi) * (c3arr[i-1] - azi) <= 0.) {
            return i;
        }
    }
    return 0;
}


int radialbounds(double rad, double apnt){
    int i;
    for (i=1; i<(NCELLS-1); i++) {
        if (c3arr[i] == apnt && c3arr[i-1] == apnt) {
            if ((c1arr[i] - rad) * (c1arr[i-1] - rad) <= 0.) {
                return i;
            }
        }
    }
    return -1;
}


int verticalbounds(double alt, double apnt, double rpnt){
    int i;
    for (i=1; i<(NCELLS-1); i++) {
        if (c3arr[i] == apnt && c1arr[i] == rpnt) {
            if (alt == 0.0) {
                return i + 1;
            }
            if (c3arr[i-1] == apnt && c1arr[i-1] == rpnt) {
                if ((c2arr[i] - alt) * (c2arr[i-1] - alt) < 0.) {
                    return i;
                }
            }
        }
    }
    return -1;
}


double findvalue(double x, double y, double z, const double arr[NCELLS]){

    // Coordinate transform.
    double rad = sqrt(x*x + y*y) / AU;
    double alt = fabs(z) / AU;
    double azi = atan2(y, x);

    // Azimuthal bounding index.
    int aidx = azimuthalbounds(azi);
    if (aidx < 0) {
        return -1.0;
    }

    // Radial bounding index.
    int ridx_a, ridx_b;
    if (aidx > 0) {
        ridx_a = radialbounds(rad, c3arr[aidx-1]);
    } else {
        ridx_a = radialbounds(rad, c3arr[NCELLS-1]);
    }
    ridx_b = radialbounds(rad, c3arr[aidx]);
    if (ridx_a < 0 || ridx_b < 0) {
        return -1.0;
    }

    // Vertical bounding index.
    int zidx_a, zidx_b, zidx_c, zidx_d;
    if (aidx > 0) {
        zidx_a = verticalbounds(alt, c3arr[aidx-1], c1arr[ridx_a-1]);
        zidx_b = verticalbounds(alt, c3arr[aidx-1], c1arr[ridx_a]);
    } else {
        zidx_a = verticalbounds(alt, c3arr[NCELLS-1], c1arr[ridx_a-1]);
        zidx_b = verticalbounds(alt, c3arr[NCELLS-1], c1arr[ridx_a]);
    }
    zidx_c = verticalbounds(alt, c3arr[aidx], c1arr[ridx_a-1]);
    zidx_d = verticalbounds(alt, c3arr[aidx], c1arr[ridx_a]);
    if (zidx_a < 0 || zidx_b < 0 || zidx_c < 0 || zidx_d < 0) {
        return -1.0;
    }

    // Interpolation.
    double val_aa = linterpolate(alt, c2arr[zidx_a-1], c2arr[zidx_a], arr[zidx_a-1], arr[zidx_a]);
    double val_ab = linterpolate(alt, c2arr[zidx_b-1], c2arr[zidx_b], arr[zidx_b-1], arr[zidx_b]);
    double val_ba = linterpolate(alt, c2arr[zidx_c-1], c2arr[zidx_c], arr[zidx_c-1], arr[zidx_c]);
    double val_bb = linterpolate(alt, c2arr[zidx_d-1], c2arr[zidx_d], arr[zidx_d-1], arr[zidx_d]);
    double val_a = linterpolate(rad, c1arr[ridx_a-1], c1arr[ridx_a], val_aa, val_ab);
    double val_b = linterpolate(rad, c1arr[ridx_b-1], c1arr[ridx_b], val_ba, val_bb);
    if (aidx > 0) {
        return linterpolate(azi, c3arr[aidx-1], c3arr[aidx], val_a, val_b);
    } else {
        return linterpolate(azi, c3arr[NCELLS-1], c3arr[aidx], val_a, val_b);
    }
}
