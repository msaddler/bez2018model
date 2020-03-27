/* This is the BEZ2018 version of the code for auditory periphery model from the Carney, Bruce and Zilany labs.
 *
 * This release implements the version of the model described in:
 *
 *   Bruce, I.C., Erfani, Y., and Zilany, M.S.A. (2018). "A Phenomenological
 *   model of the synapse between the inner hair cell and auditory nerve:
 *   Implications of limited neurotransmitter release sites," to appear in
 *   Hearing Research. (Special Issue on "Computational Models in Hearing".)
 *
 * Please cite this paper if you publish any research
 * results obtained with this code or any modified versions of this code.
 *
 * See the file readme.txt for details of compiling and running the model.
 *
 * %%% Ian C. Bruce (ibruce@ieee.org), Yousof Erfani (erfani.yousof@gmail.com),
 *     Muhammad S. A. Zilany (msazilany@gmail.com) - December 2017 %%%
 *
 * NOTE: modified by msaddler (2019-06-01) to replace MEX with Python
 * (based on https://github.com/mrkrd/cochlea/blob/master/cochlea/zilany2014)
 */

#include "Python.h"
#include "cython_bez2018.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "complex.hpp"

#define MAXSPIKES 1000000
#ifndef TWOPI
#define TWOPI 6.28318530717959
#endif

#ifndef __max
#define __max(a,b) (((a) > (b))? (a): (b))
#endif

#ifndef __min
#define __min(a,b) (((a) < (b))? (a): (b))
#endif



void SingleAN(double *px,
              double cf,
              int nrep,
              double tdres,
              int totalstim,
              double noiseType,
              double implnt,
              double spont,
              double tabs,
              double trel,
              double *meanrate,
              double *varrate,
              double *psth,
              double *synout,
              double *trd_vector,
              double *trel_vector)
{
    /* Variables for the signal-path, control-path and onward */
    double *sptime;
    double MeanISI;
    double SignalLength;
    long   MaxArraySizeSpikes;
    double tau, t_rd_rest, t_rd_init, t_rd_jump, trel_i;
    int    nSites;
    int    i, nspikes, ipst;
    double I;
    double sampFreq = 10e3; /* Sampling frequency used in the synapse */
    double total_mean_rate;
    /* Declarations of the functions used in the program */
    double Synapse(double *,
                   double,
                   double,
                   int,
                   int,
                   double,
                   double,
                   double,
                   double,
                   double *);
    int SpikeGenerator(double *,
                       double,
                       double,
                       double,
                       double,
                       double,
                       int,
                       double,
                       double,
                       double,
                       int,
                       int,
                       double,
                       long,
                       double *,
                       double *);

    /* ====== Run the synapse model ====== */
    I = Synapse(px, tdres, cf, totalstim, nrep, spont, noiseType, implnt, sampFreq, synout);

    /* Calculate the overall mean synaptic rate */
    total_mean_rate = 0;
    for(i = 0; i<I ; i++)
    {
        total_mean_rate = total_mean_rate + synout[i]/I;
    };

    /* ====== Synaptic Release/Spike Generation Parameters ====== */
    nSites = 4; /* Number of synpatic release sites */
    t_rd_rest = 14.0e-3; /* Resting value of the mean redocking time */
    t_rd_jump = 0.4e-3; /* Size of jump in mean redocking time when a redocking event occurs */
    t_rd_init = t_rd_rest+0.02e-3*spont-t_rd_jump; /* Initial value of the mean redocking time */
    tau = 60.0e-3; /* Time constant for short-term adaptation (in mean redocking time) */

    /* We register only the spikes at times after zero, the sufficient array size
     * (more than 99.7% of cases) to register spike times after zero is: */
    MeanISI = (1/total_mean_rate) + (t_rd_init)/nSites + tabs + trel;
    SignalLength = totalstim*nrep*tdres;
    MaxArraySizeSpikes= ceil ((long) (SignalLength/MeanISI + 3*sqrt(SignalLength/MeanISI)));

    sptime = (double*)calloc(MaxArraySizeSpikes,sizeof(double));
    nspikes=0;
    do {
        if (nspikes<0) /* Deal with cases where the spike time array was not long enough */
        {
            free(sptime);
            MaxArraySizeSpikes = MaxArraySizeSpikes+100; /* Make the spike time array 100 elements larger */
            sptime = (double*)calloc(MaxArraySizeSpikes,sizeof(double));
        }
        nspikes = SpikeGenerator(synout,
                                 tdres,
                                 t_rd_rest,
                                 t_rd_init,
                                 tau,
                                 t_rd_jump,
                                 nSites,
                                 tabs,
                                 trel,
                                 spont,
                                 totalstim,
                                 nrep,
                                 total_mean_rate,
                                 MaxArraySizeSpikes,
                                 sptime,
                                 trd_vector) ;
    } while (nspikes<0);  /* Repeat if spike time array was not long enough */

    /* Calculate the analytical estimates of meanrate and varrate and wrapping them up based on no. of repetitions */
    for(i = 0; i<I ; i++)
    {
        ipst = (int) (fmod(i,totalstim));
        if (synout[i]>0)
        {
            trel_i = __min(trel*100/synout[i],trel);
            trel_vector[i] = trel_i;
            /* Estimated instantaneous mean rate */
            meanrate[ipst] = meanrate[ipst] + synout[i]/(synout[i]*(tabs + trd_vector[i]/nSites + trel_i) + 1)/nrep;
            /* Estimated instananeous variance in the discharge rate */
            varrate[ipst] = varrate[ipst] + ((11*pow(synout[i],7)*pow(trd_vector[i],7))/2 + (3*pow(synout[i],8)*pow(trd_vector[i],8))/16 + 12288*pow(synout[i],2)*pow(trel_i,2) + trd_vector[i]*(22528*pow(synout[i],3)*pow(trel_i,2) + 22528*synout[i]) + pow(trd_vector[i],6)*(3*pow(synout[i],8)*pow(trel_i,2) + 82*pow(synout[i],6)) + pow(trd_vector[i],5)*(88*pow(synout[i],7)*pow(trel_i,2) + 664*pow(synout[i],5)) + pow(trd_vector[i],4)*(976*pow(synout[i],6)*pow(trel_i,2) + 3392*pow(synout[i],4)) + pow(trd_vector[i],3)*(5376*pow(synout[i],5)*pow(trel_i,2) + 10624*pow(synout[i],3)) + pow(trd_vector[i],2)*(15616*pow(synout[i],4)*pow(trel_i,2) + 20992*pow(synout[i],2)) + 12288)/(pow(synout[i],2)*pow(synout[i]*trd_vector[i] + 4,4)*(3*pow(synout[i],2)*pow(trd_vector[i],2) + 40*synout[i]*trd_vector[i] + 48)*pow(trd_vector[i]/4 + tabs + trel_i + 1/synout[i],3))/nrep;
        }
        else
            trel_vector[i] = trel;
    };

    /* Generate PSTH */
    for(i = 0; i < nspikes; i++)
    {
        ipst = (int) (fmod(sptime[i],tdres*totalstim) / tdres);
        psth[ipst] = psth[ipst] + 1;
    };
} /* End of the SingleAN function */



double Synapse(double *ihcout,
               double tdres,
               double cf,
               int totalstim,
               int nrep,
               double spont,
               double noiseType,
               double implnt,
               double sampFreq,
               double *synout)
{
    /* Initalize Variables */
    int    z, b;
    int    resamp = (int) ceil(1/(tdres*sampFreq));
    double incr = 0.0; int delaypoint = (int) floor(7500/(cf/1e3));

    double alpha1, beta1, I1, alpha2, beta2, I2, binwidth;
    int    k,j,indx,i;

    double cf_factor,cfslope,cfsat,cfconst,multFac;

    double *sout1, *sout2, *synSampOut, *powerLawIn, *mappingOut, *TmpSyn;
    double *m1, *m2, *m3, *m4, *m5;
    double *n1, *n2, *n3;

    double *randNums;
    double *sampIHC;

    mappingOut = (double*)calloc((long) ceil(totalstim*nrep),sizeof(double));
    powerLawIn = (double*)calloc((long) ceil(totalstim*nrep+3*delaypoint),sizeof(double));
    sout1 = (double*)calloc((long) ceil((totalstim*nrep+2*delaypoint)*tdres*sampFreq),sizeof(double));
    sout2 = (double*)calloc((long) ceil((totalstim*nrep+2*delaypoint)*tdres*sampFreq),sizeof(double));
    synSampOut = (double*)calloc((long) ceil((totalstim*nrep+2*delaypoint)*tdres*sampFreq),sizeof(double));
    TmpSyn = (double*)calloc((long) ceil(totalstim*nrep+2*delaypoint),sizeof(double));

    m1 = (double*)calloc((long) ceil((totalstim*nrep+2*delaypoint)*tdres*sampFreq),sizeof(double));
    m2 = (double*)calloc((long) ceil((totalstim*nrep+2*delaypoint)*tdres*sampFreq),sizeof(double));
    m3 = (double*)calloc((long) ceil((totalstim*nrep+2*delaypoint)*tdres*sampFreq),sizeof(double));
    m4 = (double*)calloc((long) ceil((totalstim*nrep+2*delaypoint)*tdres*sampFreq),sizeof(double));
    m5 = (double*)calloc((long) ceil((totalstim*nrep+2*delaypoint)*tdres*sampFreq),sizeof(double));

    n1 = (double*)calloc((long) ceil((totalstim*nrep+2*delaypoint)*tdres*sampFreq),sizeof(double));
    n2 = (double*)calloc((long) ceil((totalstim*nrep+2*delaypoint)*tdres*sampFreq),sizeof(double));
    n3 = (double*)calloc((long) ceil((totalstim*nrep+2*delaypoint)*tdres*sampFreq),sizeof(double));

    /* ================================================== */
    /* ====== Parameters of the power-law function ====== */
    /* ================================================== */
    binwidth = 1/sampFreq;
    alpha1 = 1.5e-6*100e3; beta1 = 5e-4; I1 = 0;
    alpha2 = 1e-2*100e3; beta2 = 1e-1; I2 = 0;
    /* ========================================== */
    /* ====== Generating a random sequence ====== */
    /* ========================================== */
    randNums = ffGn((int)ceil((totalstim*nrep+2*delaypoint)*tdres*sampFreq),
                    1/sampFreq,
                    0.9,
                    noiseType,
                    spont);
    /* ============================================================== */
    /* ====== Mapping function from IHCOUT to input to the PLA ====== */
    /* ============================================================== */
    cfslope = pow(spont,0.19)*pow(10,-0.87);
    cfconst = 0.1*pow(log10(spont),2)+0.56*log10(spont)-0.84;
    cfsat = pow(10,(cfslope*8965.5/1e3 + cfconst));
    cf_factor = __min(cfsat,pow(10,cfslope*cf/1e3 + cfconst))*2.0;
    multFac = __max(2.95*__max(1.0,1.5-spont/100),4.3-0.2*cf/1e3);
    k = 0;
    for (indx=0; indx<totalstim*nrep; ++indx)
    {
        mappingOut[k] = pow(10,(0.9*log10(fabs(ihcout[indx])*cf_factor))+ multFac);
        if (ihcout[indx]<0) mappingOut[k] = - mappingOut[k];
        k=k+1;
    }
    for (k=0; k<delaypoint; k++)
        powerLawIn[k] = mappingOut[0]+3.0*spont;
    for (k=delaypoint; k<totalstim*nrep+delaypoint; k++)
        powerLawIn[k] = mappingOut[k-delaypoint]+3.0*spont;
    for (k=totalstim*nrep+delaypoint; k<totalstim*nrep+3*delaypoint; k++)
        powerLawIn[k] = powerLawIn[k-1]+3.0*spont;
    /* ========================================================== */
    /* ====== Downsampling to sampFreq (low) sampling rate ====== */
    /* ========================================================== */
    sampIHC = decimate(k, powerLawIn, resamp);
    free(powerLawIn); free(mappingOut);
    /* ========================================== */
    /* ====== Running power-law adaptation ====== */
    /* ========================================== */
    k = 0;
    for (indx=0; indx<floor((totalstim*nrep+2*delaypoint)*tdres*sampFreq); indx++)
    {
        sout1[k] = __max( 0, sampIHC[indx] + randNums[indx]- alpha1*I1);
        sout2[k] = __max( 0, sampIHC[indx] - alpha2*I2);

        if (implnt==1) /* ACTUAL implementation */
        {
            I1 = 0; I2 = 0;
            for (j=0; j<k+1; ++j)
            {
                I1 += (sout1[j])*binwidth/((k-j)*binwidth + beta1);
                I2 += (sout2[j])*binwidth/((k-j)*binwidth + beta2);
            }
        } /* End of ACTUAL implementation */

        if (implnt==0) /* APPROXIMATE implementation */
        {
            if (k==0)
            {
                n1[k] = 1.0e-3*sout2[k];
                n2[k] = n1[k]; n3[0]= n2[k];
            }
            else if (k==1)
            {
                n1[k] = 1.992127932802320*n1[k-1] + 1.0e-3*(sout2[k] - 0.994466986569624*sout2[k-1]);
                n2[k] = 1.999195329360981*n2[k-1] + n1[k] - 1.997855276593802*n1[k-1];
                n3[k] = -0.798261718183851*n3[k-1] + n2[k] + 0.798261718184977*n2[k-1];
            }
            else
            {
                n1[k] = 1.992127932802320*n1[k-1] - 0.992140616993846*n1[k-2] + 1.0e-3*(sout2[k] - 0.994466986569624*sout2[k-1] + 0.000000000002347*sout2[k-2]);
                n2[k] = 1.999195329360981*n2[k-1] - 0.999195402928777*n2[k-2]+n1[k] - 1.997855276593802*n1[k-1] + 0.997855827934345*n1[k-2];
                n3[k] =-0.798261718183851*n3[k-1] - 0.199131619873480*n3[k-2]+n2[k] + 0.798261718184977*n2[k-1] + 0.199131619874064*n2[k-2];
            }
            I2 = n3[k];

            if (k==0)
            {
                m1[k] = 0.2*sout1[k];
                m2[k] = m1[k];
                m3[k] = m2[k];
                m4[k] = m3[k];
                m5[k] = m4[k];
            }
            else if (k==1)
            {
                m1[k] = 0.491115852967412*m1[k-1] + 0.2*(sout1[k] - 0.173492003319319*sout1[k-1]);
                m2[k] = 1.084520302502860*m2[k-1] + m1[k] - 0.803462163297112*m1[k-1];
                m3[k] = 1.588427084535629*m3[k-1] + m2[k] - 1.416084732997016*m2[k-1];
                m4[k] = 1.886287488516458*m4[k-1] + m3[k] - 1.830362725074550*m3[k-1];
                m5[k] = 1.989549282714008*m5[k-1] + m4[k] - 1.983165053215032*m4[k-1];
            }
            else
            {
                m1[k] = 0.491115852967412*m1[k-1] - 0.055050209956838*m1[k-2]+ 0.2*(sout1[k]- 0.173492003319319*sout1[k-1]+ 0.000000172983796*sout1[k-2]);
                m2[k] = 1.084520302502860*m2[k-1] - 0.288760329320566*m2[k-2] + m1[k] - 0.803462163297112*m1[k-1] + 0.154962026341513*m1[k-2];
                m3[k] = 1.588427084535629*m3[k-1] - 0.628138993662508*m3[k-2] + m2[k] - 1.416084732997016*m2[k-1] + 0.496615555008723*m2[k-2];
                m4[k] = 1.886287488516458*m4[k-1] - 0.888972875389923*m4[k-2] + m3[k] - 1.830362725074550*m3[k-1] + 0.836399964176882*m3[k-2];
                m5[k] = 1.989549282714008*m5[k-1] - 0.989558985673023*m5[k-2] + m4[k] - 1.983165053215032*m4[k-1] + 0.983193027347456*m4[k-2];
            }
            I1 = m5[k];
        } /* End of APPROXIMATE implementation */

        synSampOut[k] = sout1[k] + sout2[k];
        k = k+1;
    } /* End of all samples */

    free(sout1);
    free(sout2);
    free(m1);
    free(m2);
    free(m3);
    free(m4);
    free(m5);
    free(n1);
    free(n2);
    free(n3);
    /* ================================================================= */
    /* ====== Upsampling to original (high 100 kHz) sampling rate ====== */
    /* ================================================================= */
    for(z=0; z<k-1; ++z)
    {
        incr = (synSampOut[z+1]-synSampOut[z])/resamp;
        for(b=0; b<resamp; ++b)
        {
            TmpSyn[z*resamp+b] = synSampOut[z]+ b*incr;
        }
    }
    for (i=0; i<totalstim*nrep; ++i)
        synout[i] = TmpSyn[i+delaypoint];

    free(synSampOut);
    free(TmpSyn);
    free(randNums);
    free(sampIHC);
    return((long) ceil(totalstim*nrep));
} /* End of the Synapse function */



/* CompareDouble function is used to replace a mexCallMatlab sort call */
int CompareDouble (const void * a, const void * b)
{
    if ( *(double*)a <  *(double*)b ) return -1;
    if ( *(double*)a == *(double*)b ) return 0;
    if ( *(double*)a >  *(double*)b ) return 1;
}



/* Pass the output of Synapse model through the Spike Generator */
int SpikeGenerator(double *synout,
                   double tdres,
                   double t_rd_rest,
                   double t_rd_init,
                   double tau,
                   double t_rd_jump,
                   int nSites,
                   double tabs,
                   double trel,
                   double spont,
                   int totalstim,
                   int nrep,
                   double total_mean_rate,
                   long MaxArraySizeSpikes,
                   double *sptime,
                   double *trd_vector)
{
    /* Initializing the variables: */
    double* preRelease_initialGuessTimeBins;
    int*    unitRateInterval;
    double* elapsed_time;
    double* previous_release_times;
    double* current_release_times;
    double* oneSiteRedock;
    double* Xsum;

    double  MeanInterEvents;
    long    MaxArraySizeEvents;

    /* Generating a vector of random numbers without using mexCallMATLAB */
    double  *randNums;
    long    randBufIndex;
    long    randBufLen;

    long    spCount; /* Total number of spikes fired */
    long    k; /* The loop starts from kInit */
    int     i, siteNo, kInit;
    double  Tref, current_refractory_period, trel_k;
    int     t_rd_decay, rd_first;

    double  previous_redocking_period, current_redocking_period;
    int     oneSiteRedock_rounded, elapsed_time_rounded ;
    double  *preReleaseTimeBinsSorted;

    preRelease_initialGuessTimeBins = (double*)calloc(nSites, sizeof(double));
    unitRateInterval = (int*)calloc(nSites, sizeof(double));
    elapsed_time = (double*)calloc(nSites, sizeof(double));
    previous_release_times = (double*)calloc(nSites, sizeof(double));
    current_release_times = (double*)calloc(nSites, sizeof(double));
    oneSiteRedock = (double*)calloc(nSites, sizeof(double));
    Xsum = (double*)calloc(nSites, sizeof(double));

    /* Estimating Max number of spikes and events (including before zero elements) */
    MeanInterEvents = (1/total_mean_rate)+ (t_rd_init)/nSites;
    /* The sufficient array size (more than 99.7% of cases) to register event times after zero is: */
    /* MaxN=signalLengthInSec/meanEvents+ 3*sqrt(signalLengthInSec/MeanEvents) */
    MaxArraySizeEvents= ceil ((long) (totalstim*nrep*tdres/MeanInterEvents+ 3 * sqrt(totalstim*nrep*tdres/MeanInterEvents)))+nSites;

    /* Max random array Size: nSites elements for oneSiteRedock initialization, nSites elements for preRelease_initialGuessTimeBins initialization
     * 1 element for Tref initialization, MaxArraySizeSpikes elements for Tref in the loop, MaxArraySizeEvents elements one time for redocking, another time for rate intervals
     * Also, for before zero elements, Averageley add 2nSites events (redock-and unitRate) and add nSites (Max) for Trefs:
     * in total : 3 nSpikes */
    randBufLen = (long) ceil( 2*nSites+ 1 + MaxArraySizeSpikes + 2*MaxArraySizeEvents + MaxArraySizeSpikes+ 3*nSites);
    randNums = generate_random_numbers(randBufLen);
    randBufIndex = 0;

    /* Initial < redocking time associated to nSites release sites */
    for (i=0; i<nSites; i++)
    {
        oneSiteRedock[i]=-t_rd_init*log(randNums[randBufIndex++]);
    }

    /* Initial preRelease_initialGuessTimeBins associated to nsites release sites */
    for (i=0; i<nSites; i++)
    {
        preRelease_initialGuessTimeBins[i]= __max(-totalstim*nrep,ceil ((nSites/__max(synout[0],0.1) + t_rd_init)*log(randNums[randBufIndex++] ) / tdres));
    }

    /* Now sort the four initial preRelease times and associate
     * the farthest to zero as the site which has also generated a spike */
    qsort(preRelease_initialGuessTimeBins, nSites, sizeof(double), CompareDouble);
    preReleaseTimeBinsSorted = preRelease_initialGuessTimeBins;

    /* Consider the inital previous_release_times to be the preReleaseTimeBinsSorted * tdres */
    for (i=0; i<nSites; i++)
    {
        previous_release_times[i] = ((double)preReleaseTimeBinsSorted[i])*tdres;
    }

    /* The position of first spike, also where the process is started -- continued from the past */
    kInit = (int) preReleaseTimeBinsSorted[0];
    /* Current refractory time */
    Tref = tabs - trel*log( randNums[ randBufIndex++ ] );
    /* Initial refractory regions */
    current_refractory_period = (double) kInit*tdres;
    spCount = 0; /* Total number of spikes fired */
    k = kInit;  /* The loop starts from kInit */

    /* Set dynamic mean redocking time to initial mean redocking time */
    previous_redocking_period = t_rd_init;
    current_redocking_period = previous_redocking_period;
    t_rd_decay = 1; /* Logical "true" as to whether to decay the value of current_redocking_period at the end of the time step */
    rd_first = 0; /* Logical "false" as to whether a first redocking event has occurred */

    /* A loop to find the spike times for all the totalstim*nrep */
    while (k < totalstim*nrep)
    {
        for (siteNo = 0; siteNo<nSites; siteNo++)
        {
            if ( k > preReleaseTimeBinsSorted [siteNo] )
            {
                /* Redocking times do not necessarily occur exactly at time step value -- calculate the
                 * number of integer steps for the elapsed time and redocking time */
                oneSiteRedock_rounded = (int) floor(oneSiteRedock[siteNo]/tdres);
                elapsed_time_rounded = (int) floor(elapsed_time[siteNo]/tdres);
                if ( oneSiteRedock_rounded == elapsed_time_rounded )
                {
                    /* Jump trd by t_rd_jump if a redocking event has occurred */
                    current_redocking_period = previous_redocking_period + t_rd_jump;
                    previous_redocking_period = current_redocking_period;
                    t_rd_decay = 0; /* Don't decay the value of current_redocking_period if a jump has occurred */
                    rd_first = 1; /* Flag for when a jump has first occurred */
                }
                /* To be sure that for each site, the code start from its
                 * associated previus release time :*/
                elapsed_time[siteNo] = elapsed_time[siteNo] + tdres;
            };
            /* The elapsed time passes the one time redock (the redocking is finished),
             * In this case the synaptic vesicle starts sensing the input
             * for each site integration starts after the redocking is finished for the corresponding site) */
            if ( elapsed_time[siteNo] >= oneSiteRedock [siteNo] )
            {
                Xsum[siteNo] = Xsum[siteNo] + synout[__max(0,k)] / nSites;
                /* There are nSites integrals each vesicle senses 1/nosites of the whole rate */
            }
            if ( (Xsum[siteNo] >= unitRateInterval[siteNo]) && (k >= preReleaseTimeBinsSorted [siteNo]) )
            {
                /* An event -- a release happened for the siteNo */
                oneSiteRedock[siteNo] = -current_redocking_period*log( randNums[randBufIndex++]);
                current_release_times[siteNo] = previous_release_times[siteNo] + elapsed_time[siteNo];
                elapsed_time[siteNo] = 0;
                if ( (current_release_times[siteNo] >= current_refractory_period) )
                {
                    /* A spike occured for the current event -- release
                     * spike_times[(int)(current_release_times[siteNo]/tdres)-kInit+1 ] = 1; */
                    /* Register only non-negative spike times */
                    if (current_release_times[siteNo] >= 0)
                    {
                        sptime[spCount] = current_release_times[siteNo]; spCount = spCount + 1;
                    }
                    trel_k = __min(trel*100/synout[__max(0,k)],trel);
                    Tref = tabs-trel_k*log( randNums[randBufIndex++] ); /*Refractory periods */
                    current_refractory_period = current_release_times[siteNo] + Tref;
                }
                previous_release_times[siteNo] = current_release_times[siteNo];
                Xsum[siteNo] = 0;
                unitRateInterval[siteNo] = (int) (-log(randNums[randBufIndex++]) / tdres);
            };
            /* Error Catching */
            if ( (spCount+1)>MaxArraySizeSpikes || (randBufIndex+1)>randBufLen )
            {
                /* mexPrintf (" Array for spike times or random Buffer length not large enough, re-running the function."); */
                spCount = -1;
                k = totalstim*nrep;
                siteNo = nSites;
            }
        };

        /* Decay the adapative mean redocking time towards the resting value if no redocking events occurred in this time step */
        if ( (t_rd_decay==1) && (rd_first==1) )
        {
            current_redocking_period = previous_redocking_period - (tdres/tau)*( previous_redocking_period-t_rd_rest );
            previous_redocking_period = current_redocking_period;
        }
        else
        {
            t_rd_decay = 1;
        }

        /* Store the value of the adaptive mean redocking time if it is within the simulation output period */
        if ( (k>=0) && (k<totalstim*nrep) )
        {
            trd_vector [k] = current_redocking_period;
        }
        k = k+1;
    };

    free(preRelease_initialGuessTimeBins);
    free(unitRateInterval);
    free(elapsed_time);
    free(previous_release_times);
    free(current_release_times);
    free(oneSiteRedock);
    free(Xsum);
    return (spCount);
} /* End of the SpikeGenerator function */
