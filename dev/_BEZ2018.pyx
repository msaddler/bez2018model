import numpy as np
from libc.stdlib cimport malloc
from . import util
import scipy.signal as dsp

cimport numpy as np

cdef extern from "stdlib.h":
    void *memcpy(void *str1, void *str2, size_t n)


cdef extern from "model_IHC.h":
    void IHCAN(
        double *px,
        double cf,
        int nrep,
        double tdres,
        int totalstim,
        double cohc,
        double cihc,
        int species,
        double *ihcout
    )
