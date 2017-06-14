// findvalue    - wrapper for the findcell and interpolation functions.
// findcell     - finds the bounding indices.
// linterpolate - linear interpolation.

void verticalbounds(double c1val, double c2, int *lidx, int *uidx){

    /*
        Find the bounding cells, lidx and uidx, of 'c2' in a vertical column
        at radius c1val. Returns lidx == uidx if not found in column.
    */

    int i = 0;
    for (i=0; i<(NCELLS-1); i=i+1) {
        if (c1arr[i] == c1val && c1arr[i-1] == c1val) {
            if ((c2arr[i]-c2)*(c2arr[i-1]-c2) < 0.) {
                *uidx = i;
                *lidx = i-1;
                return;
            }
        } else if (c1arr[i] > c1val) {
            *uidx = i-1;
            *lidx = i-1;
            break;
        }
    }
    return;
}

double projected_value(double c1, double c2, int aidx, int cidx){

    /*
        Calculate the projection of the z value on the outer column.
        If it fails, return -1.
    */

    double z_proj;
    if (c1 == c1arr[aidx]) {
        return -1.0;
    }
    z_proj = c1arr[cidx] - c1arr[aidx];
    z_proj *= c2 - c2arr[aidx];
    z_proj /= c1 - c1arr[aidx];
    return z_proj + c2arr[aidx];
}

void findcell(double c1, double c2, int *aidx, int *bidx, int *cidx, int *didx){

    /*
        For a given position (c1, c2), find the four cells in the input model
        which bound it for bilinera interpolation. Three cases can arise:
            (1) interpolation is possible; all returned indicies are different.
            (2) bodged interpolation is possible; aidx == bidx.
            (3) interpolation is not possible; aidx = -1.
    */

    double c1lower, c1upper;
    double ylim, m, c;
    int atemp, btemp, ctemp, dtemp, zmax;
    int i = 0;

    // Find the bounding radial points such that c1lower < r <= c1upper.
    i = 0;
    while (c1arr[i] < c1) {
        if (i == NCELLS - 1) {
            *aidx = -1;
            return;
        }
        i++;
    }
    c1lower = c1arr[i-1];
    c1upper = c1arr[i];

    /*
        First attempt at finding the bounding cells.
    */

    atemp = -1;
    btemp = -1;
    ctemp = -1;
    dtemp = -1;
    verticalbounds(c1lower, c2, &atemp, &btemp);
    verticalbounds(c1upper, c2, &ctemp, &dtemp);
    *aidx = atemp;
    *bidx = btemp;
    *cidx = ctemp;
    *didx = dtemp;

    /*
        Cycle through the possible options: (3), (1) and possible (2).
    */

    if (*cidx == *didx) {
        *aidx = -1;
        return;
    } else if (*aidx != *bidx) {
        return;
    } else {

        /*
            To check if this is viable, first calculated the maximum value
            of the outer column. This allows one to calculate the line between
            the two maximum values (with the first given by aidx = bidx). We can
            then check if the value is less than this.
            If so, then project the z value onto the outer column and have
            cidx and didx being the cells which bound this value. An altered
            interpolation method can then be used.
        */

        i = 0;
        for (i=0; i<NCELLS-1; i=i+1){
            if (c1arr[i] > c2){
                break;
            }
        }
        if (i == 0) {
            *aidx = -1;
            return;
        } else if (c1arr[i] == c2) {
            zmax = i;
        } else if (c1arr[i] > c2) {
            zmax = i - 1;
        } else {
            *aidx = -1;
            return;
        }

        m = c2arr[zmax] - c2arr[*aidx];
        m /= c1arr[zmax] - c1arr[*aidx];
        c = c2arr[zmax] - m * c1arr[zmax];

        if (c2 > (m * c1 + c)) {
            *aidx = -1;
            return;
        }

    }

    /*
        Find the bounding cells of the projected value and return.
    */

    double zproj;
    zproj = projected_value(c1, c2, atemp, ctemp);
    verticalbounds(c1upper, zproj, &ctemp, &dtemp);
    if (ctemp == dtemp) {
        *aidx = -1;
        return;
    } else {
        *cidx = ctemp;
        *didx = dtemp;
        return;
    }
}


double interpolation(double c1, double c2, int aidx, int bidx, int cidx,
                     int didx, const double arr[NCELLS]) {

    /*
        Interpolation routines.
        If aidx != bidx, then do normal bilinear interpolation, otherwise,
        do bodged interpolation.
    */

    double A, B, f;
    if (aidx != bidx) {
        f = (c2 - c2arr[aidx]) / (c2arr[bidx] - c2arr[aidx]);
        A = arr[aidx] * (1. - f) + arr[bidx] * f;
        f = (c2 - c2arr[cidx]) / (c2arr[didx] - c2arr[cidx]);
        B = arr[cidx] * (1. - f) + arr[didx] * f;
        f = (c1 - c1arr[aidx]) / (c1arr[cidx] - c1arr[aidx]);
        return A * (1. - f) + B * f;
    } else {
        A = projected_value(c1, c2, aidx, cidx);
        f = (A - c2arr[cidx]) / (c2arr[didx] - c2arr[cidx]);
        B = arr[cidx] * (1. - f) + arr[didx] * f;
        f = (c1 - c1arr[aidx]) / (c1arr[cidx] - c1arr[aidx]);
        return arr[aidx] * (1. - f) + B * f;
    }
}

double findvalue(double x, double y, double z, const double arr[NCELLS]){

    /*
        Returns the interpolated value of (rad, alt). If interpolation is not
        possible then returns -1. This should be caught in the phyiscal
        properties function.
    */

    double rad = sqrt(x*x + y*y) / AU;
    double alt = fabs(z) / AU;
    double azi = atan2(y, x);
    int aidx, bidx, cidx, didx;

    findcell(rad, alt, &aidx, &bidx, &cidx, &didx);
    if (aidx >= 0) {
        return interpolation(c1, c2, aidx, bidx, cidx, didx, arr);
    } else {
        return -1.;
    }
}
