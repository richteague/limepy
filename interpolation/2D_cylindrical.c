/*
    findvalue(x, y, param) - returns interpolated value at (x, y).
    radialbounds(rad) - finds the bounding vertical slices.
    verticalbounds(alt, rpnt) - finds the bounding cells.
    linterpolate(x, x0, x1, y0, y1) - interpolates values.

    This linearally interpolates the provided grid. There is no special
    handling for points at the top of the cell. Given that this should be a
    very tenuous region in the disk it should not matter to the line emission.
*/

double linterpolate(double x, double xa, double xb, double ya, double yb){
    return (x - xa) * (yb - ya) / (xb - xa) + ya;
}


int radialbounds(double rad){
    int i;
    for (i=1; i<(NCELLS-1); i++) {
        if ((c1arr[i] - rad) * (c1arr[i-1] - rad) <= 0.) {
            return i;
        }
    }
    return -1;
}


int verticalbounds(double alt, double rpnt){
    int i;
    for (i=1; i<(NCELLS-1); i++) {
        if (c1arr[i] == rpnt) {
            if (alt == 0.0) {
                return i + 1;
            }
            if (c1arr[i-1] == rpnt) {
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

    // Radial bounding index.
    int ridx = radialbounds(rad);
    if (ridx < 0) {
        return -1.0;
    }
    
    // Vertical bounding indices.
    int zidx_a = verticalbounds(alt, c1arr[ridx-1]);
    int zidx_b = verticalbounds(alt, c1arr[ridx]);
    if (zidx_a < 0 || zidx_b < 0) {
        return -1.0;
    }

    // Interpolation (linear).
    double val_a = linterpolate(alt, c2arr[zidx_a-1], c2arr[zidx_a], arr[zidx_a-1], arr[zidx_a]);
    double val_b = linterpolate(alt, c2arr[zidx_b-1], c2arr[zidx_b], arr[zidx_b-1], arr[zidx_b]);
    return linterpolate(rad, c1arr[ridx-1], c1arr[ridx], val_a, val_b);

}
