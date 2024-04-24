# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 12:31:27 2022

@author: jlubbers


A collection of functions to help with the forward modeling of Sr and Mg in 
plagioclase using the finite difference approach outlined in Costa et al., (2008)
and Lubbers et al., (2022).

These should be used in conjunction with the methods outlined in
"Sr_plag_diffusion.ipynb" and "Mg_plag_diffusion.ipynb" found in the same repository

Happy modeling!
Jordan
"""
## plagioclase partition coefficient calculation
import warnings

import mendeleev
import numpy as np
import pandas as pd
import statsmodels.api as sm
from tqdm import tqdm


def plag_kd_calc(element, An, temp, method, sio2_melt=60):
    """
    calculates the partition coefficient for a given element in plagioclase based on its anorthite
    content according to the Arrhenius relationship as originally defined by Blundy and Wood (1991)

    This function gives the user an option of three experimental papers to choose from when calculating
    partition coefficient:

    Bindeman et al., 1998 = ['Li','Be','B','F','Na','Mg','Al','Si','P','Cl','K','Ca','Sc',
    'Ti','Cr','Fe','Co','Rb','Sr','Zr','Ba','Y','La','Ce','Pr','Nd','Sm','Eu','Pb']

    Nielsen et al., 2017 = ['Mg','Ti','Sr','Y','Zr','Ba','La','Ce','Pr','Nd','Pb']

    Tepley et al., 2010 = ['Sr','Rb','Ba','Pb','La','Nd','Sm','Zr','Th','Ti']

    Dohmen and Blundy (2014) = ["Li", "Na", "K", "Rb", "Cs","Mg",
                                 "Ca", "Sr", "Ba","Y","La","Ce","Pr",
                                 "Nd","Sm","Eu","Gd","Tb","Dy","Ho",
                                 "Er","Tm","Yb","Lu",]


    Inputs:
    -------
    element : string
    The element you are trying to calculate the partition coefficient for. See above for supported
    elements for each method

    An : array-like
    Anorthite content (between 0 and 1) of the plagioclase. This can be a scalar value or Numpy array

    temp: scalar
    Temperature in Celsius to calculate the partition coefficient at

    method : string
    choice of 'Bindeman', 'Nielsen', 'Tepley', 'Dohmen'. This uses then uses the Arrhenius parameters from
    Bindeman et al., 1998, Nielsen et al., 2017, or Tepley et al., 2010, Dohmen and Blundy (2014) respectively.


    sio2_melt : SiO2 wt% composition of the melt. Only valid if method = "Dohmen".

    Returns:
    --------
    kd : partition coefficient for the specified input parameters
    rtlnk : RTln(kd)
    A : slope of RTln(kd) vs X_an
    B : intercept of RTln(kd) vs X_an

    """
    if method not in ["Bindeman", "Nielsen", "Tepley", "Dohmen"]:
        raise ValueError(
            f"The value chosen for partitioning method is {method}: please choose 'Bindeman','Nielsen','Tepley', 'Dohmen'."
        )

    if method == "Dohmen":
        elements = [
            "Li",
            "Na",
            "K",
            "Rb",
            "Cs",
            "Mg",
            "Ca",
            "Sr",
            "Ba",
            "Y",
            "La",
            "Ce",
            "Pr",
            "Nd",
            "Sm",
            "Eu",
            "Gd",
            "Tb",
            "Dy",
            "Ho",
            "Er",
            "Tm",
            "Yb",
            "Lu",
        ]

        if element in elements:
            dohmen_kd, dohmen_rtlnk, A, B = dohmen_kd_calc(
                element, An=An, sio2_melt=sio2_melt, temp=temp
            )
        else:
            raise Exception(
                "The element you have selected is not supported by Dohmen and Blundy (2014). Please choose another one"
            )

        return dohmen_kd, dohmen_rtlnk, A, B

    else:
        if method == "Bindeman":
            # Table 4 from Bindeman et al 1998
            elements = [
                "Li",
                "Be",
                "B",
                "F",
                "Na",
                "Mg",
                "Al",
                "Si",
                "P",
                "Cl",
                "K",
                "Ca",
                "Sc",
                "Ti",
                "Cr",
                "Fe",
                "Co",
                "Rb",
                "Sr",
                "Zr",
                "Ba",
                "Y",
                "La",
                "Ce",
                "Pr",
                "Nd",
                "Sm",
                "Eu",
                "Pb",
            ]

            a = (
                np.array(
                    [
                        -6.9,
                        28.2,
                        -0.61,
                        -37.8,
                        -9.4,
                        -26.1,
                        -0.3,
                        -2,
                        -30.7,
                        -24.5,
                        -25.5,
                        -15.2,
                        -94.2,
                        -28.9,
                        -44,
                        -35.2,
                        -59.9,
                        -40,
                        -30.4,
                        -90.4,
                        -55,
                        -48.1,
                        -10.8,
                        -17.5,
                        -22.5,
                        -19.9,
                        -25.7,
                        -15.1,
                        -60.5,
                    ]
                )
                * 1e3
            )
            a_unc = (
                np.array(
                    [
                        1.9,
                        6.1,
                        0.5,
                        11.5,
                        1,
                        1.1,
                        0.8,
                        0.2,
                        4.6,
                        9.5,
                        1.2,
                        0.6,
                        28.3,
                        1.5,
                        6.3,
                        1.9,
                        10.8,
                        6.7,
                        1.1,
                        5.5,
                        2.4,
                        3.7,
                        2.6,
                        2.3,
                        4.1,
                        3.6,
                        6.3,
                        16.1,
                        11.8,
                    ]
                )
                * 1e3
            )

            b = (
                np.array(
                    [
                        -12.1,
                        -29.5,
                        9.9,
                        23.6,
                        2.1,
                        -25.7,
                        5.7,
                        -0.04,
                        -12.1,
                        11,
                        -10.2,
                        17.9,
                        37.4,
                        -15.4,
                        -9.3,
                        4.5,
                        12.2,
                        -15.1,
                        28.5,
                        -15.3,
                        19.1,
                        -3.4,
                        -12.4,
                        -12.4,
                        -9.3,
                        -9.4,
                        -7.7,
                        -14.2,
                        25.3,
                    ]
                )
                * 1e3
            )
            b_unc = (
                np.array(
                    [
                        1,
                        4.1,
                        3.8,
                        7.1,
                        0.5,
                        0.7,
                        0.4,
                        0.08,
                        2.9,
                        5.3,
                        0.7,
                        0.3,
                        18.4,
                        1,
                        4.1,
                        1.1,
                        7,
                        3.8,
                        0.7,
                        3.6,
                        1.3,
                        1.9,
                        1.8,
                        1.4,
                        2.7,
                        2.0,
                        3.9,
                        11.3,
                        7.8,
                    ]
                )
                * 1e3
            )

            plag_kd_params = pd.DataFrame(
                [a, a_unc, b, b_unc],
                columns=elements,
                index=["a", "a_unc", "b", "b_unc"],
            )

            R = 8.314

        elif method == "Nielsen":
            elements = ["Mg", "Ti", "Sr", "Y", "Zr", "Ba", "La", "Ce", "Pr", "Nd", "Pb"]
            a = (
                np.array(
                    [-10, -32.5, -25, -65.7, -25, -35.1, -32, -33.6, -29, -31, -50]
                )
                * 1e3
            )
            a_unc = (
                np.array([3.3, 1.5, 1.1, 3.7, 5.5, 4.5, 2.9, 2.3, 4.1, 3.6, 11.8]) * 1e3
            )

            b = (
                np.array([-35, -15.1, 25.5, 2.2, -50, 10, -5, -6.8, 8.7, -8.9, 22.3])
                * 1e3
            )
            b_unc = (
                np.array([2.1, 1, 0.7, 1.9, 3.6, 2.4, 2.3, 1.4, 2.7, 2.0, 7.8]) * 1e3
            )

            plag_kd_params = pd.DataFrame(
                [a, a_unc, b, b_unc],
                columns=elements,
                index=["a", "a_unc", "b", "b_unc"],
            )

            R = 8.314

        elif method == "Tepley":
            elements = ["Sr", "Rb", "Ba", "Pb", "La", "Nd", "Sm", "Zr", "Th", "Ti"]
            a = (
                np.array(
                    [
                        -50.18,
                        -35.7,
                        -78.6,
                        -13.2,
                        -93.7,
                        -84.3,
                        -108.0,
                        -70.9,
                        -58.1,
                        -30.9,
                    ]
                )
                * 1e3
            )
            a_unc = (
                np.array([6.88, 13.8, 16.1, 44.4, 12.2, 8.1, 17.54, 58.2, 35.5, 8.6])
                * 1e3
            )

            b = np.array(
                [
                    44453,
                    -20871,
                    41618,
                    -15761,
                    37900,
                    24365,
                    35372 - 7042,
                    -60465,
                    -14204,
                ]
            )
            b_unc = np.array([1303, 2437, 2964, 5484, 2319, 1492, 3106, 101886073493])

            plag_kd_params = pd.DataFrame(
                [a, a_unc, b, b_unc],
                columns=elements,
                index=["a", "a_unc", "b", "b_unc"],
            )

            if np.percentile(An, q=50) < 0.6:
                warnings.warn(
                    "Over half your An values are significantly below the calibration range in Tepley et al., (2010)"
                    "and most likely will produce partition coefficient values that are significantly overestimated",
                    stacklevel=2,
                )

            R = 8.314

        if element in elements:
            a = np.random.normal(
                plag_kd_params[element].a,
                plag_kd_params[element].a_unc,
                1000,
            )
            b = np.random.normal(
                plag_kd_params[element].b,
                plag_kd_params[element].b_unc,
                1000,
            )

            random_kds = np.exp(
                (a[:, np.newaxis] * An + b[:, np.newaxis]) / (R * temp + 273.15)
            )

            rtlnk = plag_kd_params[element].a * An + plag_kd_params[element].b
            rtlnk = rtlnk / 1000
            kd = np.exp(rtlnk * 1000 / (R * temp + 273.15))

            # kd_mean = np.mean(random_kds, axis=0)
            kd_std = np.std(random_kds, axis=0)

        else:
            raise Exception(
                "The element you have selected is not supported by this function. Please choose another one"
            )

        return kd, rtlnk, a.mean() / 1000, b.mean() / 1000


# building a time grid
def get_tgrid(iterations, timestep):
    """
    generating a time grid for the diffusion model to iterate over

    Parameters
    ----------
    iterations : int
        The number of total iterations you want the model to be
    timestep : string
        how to space the time grid. Options are "hours", "days", "months",
        "tenths","years". The time grid will be spaced by the amount of seconds
        in the specified unit effectively making a "dt"

    Returns
    -------
    t : ndarray
        time grid that starts at 0, is spaced by the number of seconds in the
        specified timestep, and is n-iterations in shape.

    """

    sinyear = 60 * 60 * 24 * 365.25
    tenthsofyear = sinyear / 10
    days = sinyear / 365.25
    months = sinyear / 12
    hours = sinyear / 8760

    if timestep == "days":
        step = days
    elif timestep == "months":
        step = months
    elif timestep == "hours":
        step = hours
    elif timestep == "tenths":
        step = tenthsofyear
    elif timestep == "years":
        step = sinyear
    # create a time grid that starts at 0
    # goes to n iterations and is spaced by
    # the desired step.
    t = np.arange(0, iterations * step + 1, step)
    return t


# diffusivity of Sr and Mg in plagioclase


def plag_diffusivity(
    element, an, T_K, method="van orman", asio2=0.55, return_vector=True
):
    """
    A function for calculating the diffusion coefficient for Sr and Mg in
    plagioclase

    Mg "van orman" uses Van orman et al., (2014)
    Mg "costa" uses Costa et al., (2003)
    Mg "faak" uses Faak et al., (2013)
    Sr uses Druitt et al., (2012) which is adapted from Giletti and Casserly
    (1994)

    Parameters
    ----------
    element : string
        "Sr" or "Mg"
    an : array-like
        anorthite value of the plagioclase in fraction An (e.g., 0-1)
    T_K : scalar
        temperature in kelvin
    method : string, optional
        which model to use for Mg diffusion coefficient. Options are either
        "costa" which uses the relationship in costa et al., 2003 or "van orman"
        which uses the relationship from van orman et al., 2014.
        The default is "van orman".
    asio2 : array-like
        silica activity. if the element == 'Mg' and method == 'faak', the
        diffusion coefficient requires that silica activity is known. Defaults to
        0.55. Must be between 0.55 and 1 to match their experiments. Changes from
        0.55 --> 1 result in a ~factor of 5 change in D
    return_vector : boolean
        whether or not to return a An-length vector of D values required for diffusion
        modeling input into `diffuse_forward` or `diffuse_forward_halfspace`. If True,
        then asio2 must be scalar.


    Returns
    -------
    D : array-like
        diffusion coefficient for specified element and model in um^2/s
        ** if 'faak' is chosen, and asio2 is scalar, then output is scalar
        because it does not depend on An values. To map to the same shape as
        your trace element profile simply choose `return_vector = True`. Else, if
        asio2 is a vector, a asio2-length vector will be returned. 


    """

    R = 8.314

    if element == "Mg":
        if method == "van orman":
            D = np.exp(-6.06 - 7.96 * an - 287e3 / (R * T_K)) * 1e12

        elif method == "costa":
            D = 2.92 * 10 ** (-4.1 * an - 3.1) * np.exp(-266000 / (R * T_K)) * 1e12

        elif method == "faak":
            assert asio2 >= 0.55
            assert asio2 <= 1.0
            D = 1.25 * 10**-4 * np.exp(-320924 / (R * T_K)) * asio2**2.6 * 1e12
            if return_vector is True:
                #
                assert isinstance(asio2,float), "to use this option asio2 must be a scalar"
                D = np.full(an.shape, D)

    if element == "Sr":
        D = 2.92 * 10 ** (-4.1 * an - 4.08) * np.exp(-276000 / (R * T_K)) * 1e12

    return D


## diffusion equation
def diffuse_forward(
    initial_profile,
    te,
    t,
    D,
    an_smooth,
    A,
    dist,
    T_K,
    boundary="infinite observed",
):
    """
    Function for running a forward diffusion model for either Sr or Mg in plagioclase
    based on the discretized solution to Eq. 7 from Costa et al., 2003

    Parameters
    ----------
    initial_profile : ndarray
        Where the forward model is starting from. i.e., the initial composition
        of your plagioclase trace element profile
    te : ndarray
        the observed (measured) trace element profile in the plagioclase
    t : ndarray
        time grid array. This is an array that is the output from the
        plag_diff.get_tgrid() function

    D : ndarray
       diffusion coefficient for each point in the profile
    an_smooth : ndarray
        anorthite fraction for each point in the profile
    A : ndarray
        value pertaining to the "A" parameter in the Arrenhius partitioning
        relationship for trace element partitioning in plagioclase as
        defined by Bindeman et al., 1998. This is part of the output for the
        plag_diff.plag_kd_calc() function
    dist : ndarray
        Array pertaining to the distance of your profile. It is te.shape[0]
        points long and spaced by the analytical resolution.
    T_K : scalar
        model temperature in Kelvin
    boundary : string, optional
        How to treat the most "rimward" boundary of the model (i.e., the point that
        is furthest to the right). Options are:

            "infinite observed": a fixed reservoir assumption where the most rimward
            point is fixed at the value that is determined by the most rimward analysis
            "infinite model": similar to infinite observed, however the value is fixed
            at the most rimward initial profile value
            "open": this does not fix the most rimward analysis and lets it diffuse
            like the rest of the points. This may be useful if you have a transect
            that is not necessarily at the rim of the grain

            default is "infinite observed"


    Returns
    -------
    curves : ndarray
        array that is t.shape[0] x distance.shape[0] and pertains to a diffusion
        curve for each timestep in the model. E.g. you can plot each diffusion curve
        like this:
            fig, ax = plt.subplots()
            ax.plot(dist, curves[0]) # will plot the first timestep in the model

    """
    # containers for each iteration
    # unknown at current iteration
    u = np.zeros(initial_profile.shape[0])
    # u at previous iteration
    u_n = initial_profile.copy()
    nx = initial_profile.shape[0]
    nt = t.shape[0]
    dt = t[1] - t[0]
    dx = dist[1] - dist[0]
    R = 8.314

    # creating a container to put all of your curve iterations
    curves = np.zeros((nt, nx))

    # iterating over the length of Nt(each iteration is a time step)
    for n in tqdm(range(0, int(nt)), total=nt, unit="timestep"):
        # this is long...
        u[1 : nx - 1] = u_n[1 : nx - 1] + dt * (
            ((D[2:nx] - D[1 : nx - 1]) / dx) * ((u_n[2:nx] - u_n[1 : nx - 1]) / dx)
            + D[1 : nx - 1]
            * ((u_n[2:nx] - 2 * u_n[1 : nx - 1] + u_n[0 : nx - 2]) / dx**2)
            - (A / (R * T_K))
            * (
                D[1 : nx - 1]
                * (
                    ((u_n[2:nx] - u_n[1 : nx - 1]) / dx)
                    * ((an_smooth[2:nx] - an_smooth[1 : nx - 1]) / dx)
                )
                + u_n[1 : nx - 1]
                * (
                    ((D[2:nx] - D[1 : nx - 1]) / dx)
                    * ((an_smooth[2:nx] - an_smooth[1 : nx - 1]) / dx)
                )
                + D[1 : nx - 1]
                * u_n[1 : nx - 1]
                * (
                    (
                        an_smooth[2:nx]
                        - 2 * an_smooth[1 : nx - 1]
                        + an_smooth[0 : nx - 2]
                    )
                    / dx**2
                )
            )
        )
        # letting the most 'core-ward' boundary condition diffuse according to the Costa 2003 equation
        u[0] = u_n[0] + dt * (
            ((D[1] - D[0]) / dx) * ((u_n[1] - u_n[0]) / dx)
            + D[0] * ((u_n[1] - 2 * u_n[0] + u_n[1]) / dx**2)
            - (A / (R * T_K))
            * (
                D[0] * (((u_n[1] - u_n[0]) / dx) * ((an_smooth[1] - an_smooth[0]) / dx))
                + u_n[0] * (((D[1] - D[0]) / dx) * ((an_smooth[1] - an_smooth[0]) / dx))
                + D[0]
                * u_n[0]
                * ((an_smooth[1] - 2 * an_smooth[0] + an_smooth[1]) / dx**2)
            )
        )

        if boundary == "infinite observed":
            # fix the most 'rim-ward' concentration (infinite reservoir assumption) based on observed data
            u[-1] = te[-1]

        elif boundary == "infinite model":
            # infinite reservoir based on boundary condition fixed based on boundary conditions
            u[-1] = initial_profile[-1]

        elif boundary == "open":
            # not infinite reservoir assumption. Let's everything diffuse and does not keep it fixed
            # potentially useful for grains that are not at the rim
            u[-1] = u_n[-1] + dt * (
                ((D[-2] - D[-1]) / dx) * ((u_n[-2] - u_n[-1]) / dx)
                + D[-1] * ((u_n[-2] - 2 * u_n[-1] + u_n[-2]) / dx**2)
                - (A / (R * T_K))
                * (
                    D[-1]
                    * (
                        ((u_n[-2] - u_n[-1]) / dx)
                        * ((an_smooth[-2] - an_smooth[-1]) / dx)
                    )
                    + u_n[-1]
                    * (((D[-2] - D[-1]) / dx) * ((an_smooth[-2] - an_smooth[-1]) / dx))
                    + D[-1]
                    * u_n[-1]
                    * ((an_smooth[-2] - 2 * an_smooth[-1] + an_smooth[-2]) / dx**2)
                )
            )

        # saving your iteration to your curve container
        curves[n, :] = u
        # switch your variables before the next iteration
        # makes your current u vals the u_n vals in the next loop
        u_n[:] = u

    return curves


# fitting the model using chi squared
def fit_model(te, curves):
    """
    Find the best fit timestep for the diffusion model that matches the
    observed data. Uses a standard chi-squared goodness of fit test.

    Parameters
    ----------
    te : ndarray
        the observed (measured) trace element profile in the plagioclase
    curves : ndarray
        array that is t.shape[0] x distance.shape[0] and pertains to a diffusion
        curve for each timestep in the model.

    Returns
    -------
    bf_time : int
       the best fit iteration of the model. Can be plotted as follows:

           fig,ax = plt.subplots()
           ax.plot(dist,curves[bf_time])

    """

    chi2 = abs(np.sum((te[None, :] - curves) ** 2 / (te[None, :]), axis=1))

    # find the minimum value
    chi2_min = np.min(chi2)

    # find where in the array it is (e.g., it's position)
    fit_idx = np.argwhere(chi2 == chi2_min)

    # Get that array index
    fit_idx = fit_idx[0].item()

    # add one because python starts counting at 0
    bf_time = fit_idx + 1
    return bf_time, chi2


# random profile generator for the monte carlo simulation
def random_profile(y, yerr):
    """
    Generate a random profile based on the analytical uncertainty and mean
    value at each point in the profile

    Parameters
    ----------

    y : ndarray
        array pertaining to the mean values of your trace element profile.
        i.e. your observed data
    yerr : ndarray
        array in same shape as y pertaining to the one sigma uncertainty of
        the analyses

    Returns
    -------
    yrand : ndarray
        array in same shape as y but each point in the profile is a normally
        distributed random point based on the mean and standard deviation at
        that point

    """

    # np.random.normal(mean,std deviation)
    yrand = np.random.normal(loc=y, scale=yerr)
    return yrand


def Monte_Carlo_FD(
    initial_profile,
    te,
    te_unc,
    t,
    D,
    an_smooth,
    A,
    dist,
    T_K,
    n,
    limit,
    boundary="infinite observed",
    local_minima=False,
):
    """


    Parameters
    ----------
    initial_profile : ndarray
        Where the forward model is starting from. i.e., the initial composition
        of your plagioclase trace element profile
    te : ndarray
        the observed (measured) trace element profile in the plagioclase
    t : ndarray
        time grid array. This is an array that is the output from the
        plag_diff.get_tgrid() function

    D : ndarray
       diffusion coefficient for each point in the profile
    an_smooth : ndarray
        anorthite fraction for each point in the profile
    A : ndarray
        value pertaining to the "A" parameter in the Arrenhius partitioning
        relationship for trace element partitioning in plagioclase as
        defined by Bindeman et al., 1998. This is part of the output for the
        plag_diff.plag_kd_calc() function
    dist : ndarray
        Array pertaining to the distance of your profile. It is te.shape[0]
        points long and spaced by the analytical resolution.
    T_K : scalar
        model temperature in Kelvin
    n : int
       number of iterations in the monte carlo simulation
    limit : the maximum duration for each iteration to search for the best fit
            time
    boundary : string, optional
        How to treat the most "rimward" boundary of the model (i.e., the point that
        is furthest to the right). Options are:

            "infinite observed": a fixed reservoir assumption where the most rimward
            point is fixed at the value that is determined by the most rimward analysis
            "infinite model": similar to infinite observed, however the value is fixed
            at the most rimward initial profile value
            "open": this does not fix the most rimward analysis and lets it diffuse
            like the rest of the points. This may be useful if you have a transect
            that is not necessarily at the rim of the grain

            default is "infinite observed"
    local_minima : Boolean, optional
        Whether or not there are local minima in the original model goodness
        of fit vs time plot. Usually a diffusion model converges towards a
        global minima where goodness of fit is minimized and then increases
        indefinitely afterwards. To decrease computation time and capitalize on this
        the monte carlo simulation will run until it finds the best fit and assume
        that all models after this do not fit the observed data as well. If
        you believe this does not apply choose True, where each iteration of the
        monte carlo simulation runs for "limit" number of times before proceeding
        to the next iteration. This takes significantly longer. The default is False.

    Returns
    -------
    best_fits : ndarray
        array of best fit iterations. Each value in the array is analagous
        to the "bf_time" output from plag.diffuse_forward(). It will be n
        values long.

    """

    if local_minima is False:
        best_fits = []
        # containers for each iteration
        # unknown at current iteration
        # containers for each iteration
        # unknown at current iteration
        u = np.zeros(initial_profile.shape[0])
        # u at previous iteration
        u_n = initial_profile.copy()
        nx = initial_profile.shape[0]
        nt = t.shape[0]
        dt = t[1] - t[0]
        dx = dist[1] - dist[0]
        R = 8.314
        # creating a container to put all of your curve iterations
        curves = np.zeros((nt, nx))

        for i in tqdm(range(0, n)):
            yrand = random_profile(te, te_unc)

            chi2_p = 100000
            chi2_c = 99999
            count = 0

            u = np.zeros(initial_profile.shape[0])
            # u at previous iteration
            u_n = initial_profile.copy()

            while chi2_c < chi2_p:
                count += 1
                chi2_p = chi2_c
                # this is long...
                # this is long...
                u[1 : nx - 1] = u_n[1 : nx - 1] + dt * (
                    ((D[2:nx] - D[1 : nx - 1]) / dx)
                    * ((u_n[2:nx] - u_n[1 : nx - 1]) / dx)
                    + D[1 : nx - 1]
                    * ((u_n[2:nx] - 2 * u_n[1 : nx - 1] + u_n[0 : nx - 2]) / dx**2)
                    - (A / (R * T_K))
                    * (
                        D[1 : nx - 1]
                        * (
                            ((u_n[2:nx] - u_n[1 : nx - 1]) / dx)
                            * ((an_smooth[2:nx] - an_smooth[1 : nx - 1]) / dx)
                        )
                        + u_n[1 : nx - 1]
                        * (
                            ((D[2:nx] - D[1 : nx - 1]) / dx)
                            * ((an_smooth[2:nx] - an_smooth[1 : nx - 1]) / dx)
                        )
                        + D[1 : nx - 1]
                        * u_n[1 : nx - 1]
                        * (
                            (
                                an_smooth[2:nx]
                                - 2 * an_smooth[1 : nx - 1]
                                + an_smooth[0 : nx - 2]
                            )
                            / dx**2
                        )
                    )
                )
                # letting the most 'core-ward' boundary condition diffuse according to the Costa 2003 equation
                u[0] = u_n[0] + dt * (
                    ((D[1] - D[0]) / dx) * ((u_n[1] - u_n[0]) / dx)
                    + D[0] * ((u_n[1] - 2 * u_n[0] + u_n[1]) / dx**2)
                    - (A / (R * T_K))
                    * (
                        D[0]
                        * (
                            ((u_n[1] - u_n[0]) / dx)
                            * ((an_smooth[1] - an_smooth[0]) / dx)
                        )
                        + u_n[0]
                        * (((D[1] - D[0]) / dx) * ((an_smooth[1] - an_smooth[0]) / dx))
                        + D[0]
                        * u_n[0]
                        * ((an_smooth[1] - 2 * an_smooth[0] + an_smooth[1]) / dx**2)
                    )
                )

                if boundary == "infinite observed":
                    # fix the most 'rim-ward' concentration (infinite reservoir assumption) based on observed data
                    u[-1] = te[-1]

                elif boundary == "infinite model":
                    # infinite reservoir based on boundary condition fixed based on boundary conditions
                    u[-1] = initial_profile[-1]

                else:
                    # not infinite reservoir assumption. Let's everything diffuse and does not keep it fixed
                    # potentially useful for grains that are not at the rim
                    u[-1] = u_n[-1] + dt * (
                        ((D[-2] - D[-1]) / dx) * ((u_n[-2] - u_n[-1]) / dx)
                        + D[-1] * ((u_n[-2] - 2 * u_n[-1] + u_n[-2]) / dx**2)
                        - (A / (R * T_K))
                        * (
                            D[-1]
                            * (
                                ((u_n[-2] - u_n[-1]) / dx)
                                * ((an_smooth[-2] - an_smooth[-1]) / dx)
                            )
                            + u_n[-1]
                            * (
                                ((D[-2] - D[-1]) / dx)
                                * ((an_smooth[-2] - an_smooth[-1]) / dx)
                            )
                            + D[-1]
                            * u_n[-1]
                            * (
                                (an_smooth[-2] - 2 * an_smooth[-1] + an_smooth[-2])
                                / dx**2
                            )
                        )
                    )

                # saving your iteration to your curve container
                curves[n, :] = u
                # switch your variables before the next iteration
                # makes your current u vals the u_n vals in the next loop
                u_n[:] = u

                chi2_c = np.sum(
                    (u - yrand) ** 2 / yrand,
                )

                if count == limit:
                    break
            bf_time_mc = count
            best_fits.append(bf_time_mc)
        best_fits = np.array(best_fits)

    elif local_minima is True:
        best_fits = []

        for i in tqdm(range(0, n), total=n, unit="diffusion model"):
            yrand = random_profile(dist, te, te_unc)
            # containers for each iteration
            # unknown at current iteration
            u = np.zeros(initial_profile.shape[0])
            # u at previous iteration
            u_n = initial_profile.copy()
            # creating a container to put all of your curve iterations
            curves = np.zeros((nt, nx))

            # iterating over the length of Nt(each iteration is a time step)
            for n in range(0, int(nt)):
                # this is long...
                u[1 : nx - 1] = u_n[1 : nx - 1] + dt * (
                    ((D[2:nx] - D[1 : nx - 1]) / dx)
                    * ((u_n[2:nx] - u_n[1 : nx - 1]) / dx)
                    + D[1 : nx - 1]
                    * ((u_n[2:nx] - 2 * u_n[1 : nx - 1] + u_n[0 : nx - 2]) / dx**2)
                    - (A / (R * T_K))
                    * (
                        D[1 : nx - 1]
                        * (
                            ((u_n[2:nx] - u_n[1 : nx - 1]) / dx)
                            * ((an_smooth[2:nx] - an_smooth[1 : nx - 1]) / dx)
                        )
                        + u_n[1 : nx - 1]
                        * (
                            ((D[2:nx] - D[1 : nx - 1]) / dx)
                            * ((an_smooth[2:nx] - an_smooth[1 : nx - 1]) / dx)
                        )
                        + D[1 : nx - 1]
                        * u_n[1 : nx - 1]
                        * (
                            (
                                an_smooth[2:nx]
                                - 2 * an_smooth[1 : nx - 1]
                                + an_smooth[0 : nx - 2]
                            )
                            / dx**2
                        )
                    )
                )
                # letting the most 'core-ward' boundary condition diffuse according to the Costa 2003 equation
                u[0] = u_n[0] + dt * (
                    ((D[1] - D[0]) / dx) * ((u_n[1] - u_n[0]) / dx)
                    + D[0] * ((u_n[1] - 2 * u_n[0] + u_n[1]) / dx**2)
                    - (A / (R * T_K))
                    * (
                        D[0]
                        * (
                            ((u_n[1] - u_n[0]) / dx)
                            * ((an_smooth[1] - an_smooth[0]) / dx)
                        )
                        + u_n[0]
                        * (((D[1] - D[0]) / dx) * ((an_smooth[1] - an_smooth[0]) / dx))
                        + D[0]
                        * u_n[0]
                        * ((an_smooth[1] - 2 * an_smooth[0] + an_smooth[1]) / dx**2)
                    )
                )

                if boundary == "infinite observed":
                    # fix the most 'rim-ward' concentration (infinite reservoir assumption) based on observed data
                    u[-1] = te[-1]

                elif boundary == "infinite model":
                    # infinite reservoir based on boundary condition fixed based on boundary conditions
                    u[-1] = initial_profile[-1]

                else:
                    # not infinite reservoir assumption. Let's everything diffuse and does not keep it fixed
                    # potentially useful for grains that are not at the rim
                    u[-1] = u_n[-1] + dt * (
                        ((D[-2] - D[-1]) / dx) * ((u_n[-2] - u_n[-1]) / dx)
                        + D[-1] * ((u_n[-2] - 2 * u_n[-1] + u_n[-2]) / dx**2)
                        - (A / (R * T_K))
                        * (
                            D[-1]
                            * (
                                ((u_n[-2] - u_n[-1]) / dx)
                                * ((an_smooth[-2] - an_smooth[-1]) / dx)
                            )
                            + u_n[-1]
                            * (
                                ((D[-2] - D[-1]) / dx)
                                * ((an_smooth[-2] - an_smooth[-1]) / dx)
                            )
                            + D[-1]
                            * u_n[-1]
                            * (
                                (an_smooth[-2] - 2 * an_smooth[-1] + an_smooth[-2])
                                / dx**2
                            )
                        )
                    )

                # saving your iteration to your curve container
                curves[n, :] = u
                # switch your variables before the next iteration
                # makes your current u vals the u_n vals in the next loop
                u_n[:] = u

            # curves = diffuse_forward(initial_profile, boundary= boundary)

            chi2 = abs(
                np.sum((yrand[None, :] - curves) ** 2 / (yrand[None, :]), axis=1)
            )
            chi2_min = np.min(chi2)
            fit_idx = np.argwhere(chi2 == chi2_min)
            fit_idx = fit_idx[0].item()
            bf_time_mc = fit_idx + 1

            best_fits.append(bf_time_mc)

        best_fits = np.array(best_fits)

    return best_fits


# tranforming monte carlo distribution to normal
def transform_data(x, kind="log"):
    """
    transform the monte carlo data to fit a certain distribution if it is not
    normal. Standard deviations only have predictive power if the data are
    normally distributed. If the results of the monte carlo simulation are
    not normally distributed, this will transform them using either a log or
    square root transform, take the mean, median, and standard deviation and
    back transform those values into the orignal units and return them.

    Parameters
    ----------
    x : ndarray
        array of best fit iterations i.e. the output from the plag.Monte_Carlo_FD()
        function
    kind : string, optional
        the type of distribution you believe your data are. Options are "sqrt" or
        "log" and correspond to square root and log transformations respectively.
        The default is 'log'.

    Returns
    -------
    transform : ndarray
        transformed values of x
    back_mean : scalar
        back transformed mean of x
    back_median : scalar
        back transformed median of x
    back_std_l : scalar
        back transformed lower 2 sigma standard deviation of x
    back_std_u : scalar
        back transformed upper 2 sigma standard deviation of x

    """

    if kind == "sqrt":
        # Transforming your data to make it normally distributed
        transform = np.sqrt(x)
        transform_std = np.std(transform)
        transform_mean = np.mean(transform)
        transform_median = np.median(transform)

        # Back calculate mean and standard deviation
        back_mean = transform_mean**2
        back_median = transform_median**2
        back_std_l = (transform_mean - 2 * transform_std) ** 2
        back_std_u = (transform_mean + 2 * transform_std) ** 2

    if kind == "log":
        # Transforming your data to make it normally distributed
        transform = np.log(x)
        transform_std = np.std(transform)
        transform_mean = np.mean(transform)
        transform_median = np.median(transform)

        # Back calculate mean and standard deviation
        back_mean = np.exp(transform_mean)
        back_median = np.exp(transform_median)
        back_std_l = np.exp(transform_mean - 1.96 * transform_std)
        back_std_u = np.exp(transform_mean + 1.96 * transform_std)

    return transform, back_mean, back_median, back_std_l, back_std_u


def dohmen_kd_calc_lepr(element, An, sio2_melt, temp):
    T_K = temp + 273.15  # K
    R = 8.314  # J/molK

    # calculate partition coefficients of Ca and Na based on LEPR database
    # these are equations 28a-b
    Kp_ca = np.exp((-19107 + 467 * sio2_melt) / (R * T_K))
    Kp_na = np.exp((13621 - 18990 * An) / (R * T_K))

    # Eq 27 for La partition coefficient based on Ca and Na
    Kp_la = Kp_ca**2 / Kp_na * np.exp((4400 - 30.8 * T_K) / (R * T_K))

    # get the ionic radius for 8 fold coordination and +1 charge
    if element in ["Li", "Na", "K", "Rb", "Cs"]:
        for i in mendeleev.element("Na").ionic_radii:
            if (i.coordination == "VIII") and (i.charge == 1):
                r_ref = i.ionic_radius / 100  # convert from pm to Angstroms

        for i in mendeleev.element(element).ionic_radii:
            if (i.coordination == "VIII") and (i.charge == 1):
                r_i = i.ionic_radius / 100  # convert from pm to Angstroms

        # Eq 19a for +1 charge
        r_m = 1.237 + -0.0171 * An
        # Eq 19b for +1 charge
        E_m = 49.05 + 17.16 * An

        # this is just so we can use the same general form for the Kd calc at the end
        Kp = Kp_na

    # get the ionic radius for 8 fold coordination and +2 charge
    elif element in ["Mg", "Ca", "Sr", "Ba"]:
        for i in mendeleev.element("Ca").ionic_radii:
            if (i.coordination == "VIII") and (i.charge == 2):
                r_ref = i.ionic_radius / 100  # convert from pm to Angstroms

        for i in mendeleev.element(element).ionic_radii:
            if (i.coordination == "VIII") and (i.charge == 2):
                r_i = i.ionic_radius / 100  # convert from pm to Angstroms

        # Eq 19b for +2 charge
        E_m = 120.0382 + -0.3686 * An
        # Eq 22 which is a modification of Eq 19 for the +2 cations
        r_m = 1.2895 + 0.00013 * (T_K - 1563) + (-0.0952 + -0.00004 * (T_K - 1563)) * An

        # this is just so we can use the same general form at the end
        Kp = Kp_ca

    elif element in [
        "Y",
        "La",
        "Ce",
        "Pr",
        "Nd",
        "Sm",
        "Eu",
        "Gd",
        "Tb",
        "Dy",
        "Ho",
        "Er",
        "Tm",
        "Yb",
        "Lu",
    ]:
        # get the ionic radius for 8 fold coordination and +3 charge
        for i in mendeleev.element("La").ionic_radii:
            if (i.coordination == "VIII") and (i.charge == 3):
                r_ref = i.ionic_radius / 100  # convert from pm to Angstroms

        for i in mendeleev.element(element).ionic_radii:
            if (i.coordination == "VIII") and (i.charge == 3):
                r_i = i.ionic_radius / 100  # convert from pm to Angstroms

        # Eq 19a for +3 charge
        r_m = 1.331 + -0.068 * An
        # Eq 19b for +3 charge
        E_m = 152 + -31 * An

        # this is just so we can use the same general form at the end
        Kp = Kp_la

    # General form for Eqs 18a-c based on the customized parameters in the conditional
    # statements above for different cation charges.
    dohmen_kd = Kp * np.exp(
        -910.17
        * E_m
        / T_K
        * (r_m / 2 * (r_ref**2 - r_i**2) - 1 / 3 * (r_ref**3 - r_i**3))
    )
    dohmen_rtlnk = R * T_K * np.log(dohmen_kd) / 1000

    return dohmen_kd, dohmen_rtlnk


def dohmen_kd_calc(element, An, sio2_melt, temp):
    """calculate the partition coefficient of either Sr, Mg, or Ba
            in plagioclase according to the thermodynamic model outlined in
            Dohmen and Blundy (2014) doi: 10.2475/09.2014.04

            where partition coefficients for Ca and Na are derived from
            using the LEPR database (Eqs 28a-b).

            Will also calculate A and B parameters required for modeling
            diffusion of Sr and Mg as outlined in Costa et al. (2003)
            doi: 10.1016/S0016-7037(02)01345-5 which is effectively
            a regression of RTln(Kd) vs X_An where A is the slope
            and B is the intercept.


    Args:
        element (str): element to calculate partition coefficient for

        An (array-like): fraction anorthite content of the plagioclase (X_an)

        sio2_melt (array-like): SiO2 wt% composition of the melt

        temp (array-like): temperature in degrees C

    Returns:
        dohmen_kd : partition coefficient
        dohmen_rtlnk : RTln(dohmen_k) in kJ/mol
        A : slope of regression in dohmen_rtlnk vs X_an space
        B : intercept of regression in dohmen_rtlnk vs X_an space
    """
    dohmen_kd, dohmen_rtlnk = dohmen_kd_calc_lepr(
        element=element, An=An, sio2_melt=sio2_melt, temp=temp
    )
    x_an = np.linspace(0, 1, 101)
    x = x_an.copy()
    x = sm.add_constant(x)
    model = sm.OLS(
        dohmen_kd_calc_lepr(element=element, An=x_an, sio2_melt=sio2_melt, temp=temp)[
            1
        ],
        x,
    )
    result = model.fit()
    B, A = result.params

    return dohmen_kd, dohmen_rtlnk, A, B


def mutch_kd_calc(An, temp, sio2_melt):
    """Calculate the partition coefficient for Mg in plagioclase
        according to Mutch et al 2022 doi: 10.1016/j.gca.2022.10.035

    Args:
        An (array-like): fraction anorthite content of the plagioclase (X_an)

        sio2_melt (array-like): SiO2 wt% composition of the melt

        temp (array-like): temperature in degrees C

    Returns:
        kd_mutch : partition coefficient for Mg in plagioclase
        rtlnk_mutch : RTln(kd_mutch) in kj/mol
        x_an_dev : X_An - X_C1I1...the difference between the observed
        anorthite value and the anorthite value of the C1-I1 transition
        at the given temperature.
    """
    lower, upper = 10, 90
    T_K = temp + 273.15
    R = 8.314
    x_an_dev = An - (0.12 + 0.00038 * T_K)
    beta = np.zeros(x_an_dev.shape)
    beta[x_an_dev > 0] = 1

    a1 = 16.9
    a2 = -37.2
    a3 = 0.83
    a4 = -83.3

    rtlnk_mutch = (a1 + a2 * beta) * x_an_dev + a3 * sio2_melt + a4
    rtlnk_mutch = rtlnk_mutch
    kd_mutch = np.exp(rtlnk_mutch / ((R / 1000) * T_K))
    mutch_A = (
        (
            rtlnk_mutch[np.where(An >= np.percentile(An, upper))[0][0]]
            - rtlnk_mutch[np.where(An <= np.percentile(An, lower))[0][0]]
        )
        / (
            An[np.where(An >= np.percentile(An, upper))[0][0]]
            - An[np.where(An <= np.percentile(An, lower))[0][0]]
        )
        * 1000
    )

    return kd_mutch, rtlnk_mutch, mutch_A / 1000, x_an_dev


def create_stepped_profile(
    dist, step_start, step_stop, step_left, step_middle, step_right
):
    """Create a stepped profile (1D array) where the height, width, and number of steps are
    user specified

    Args:
        dist (array-like): 1D array corresponding to the distance along the measured profile
        step_start (list): list of `dist` values where each step function should start
        step_stop (list): list of `dist` values where each step function should stop
        step_left (list): list of values that correspond to the concentration on the left
        side of the step function
        step_middle (list): list of values that correspond to the concentration in the middle of
        the step function
        step_right (list): list of values that correspond to the concentration on the right
        side of the step function

    Returns:
        stepped_profile : 1D array that has step functions described by `step_start`,`step_stop`,
        `step_left`, `step_middle`, `step_right`.
    """

    stepped_profile = np.zeros(dist.shape[0])
    step_begin_idxs = []
    step_end_idxs = []

    dx = dist[1] - dist[0]

    for i in range(len(step_start)):
        stepstart = step_start[i] - np.min(dist)
        step_begin = stepstart
        step_begin_idx = int(step_begin / dx)
        step_begin_idxs.append(step_begin_idx)

        stepstop = step_stop[i] - np.min(dist)
        step_end = stepstop
        step_end_idx = int(step_end / dx)
        step_end_idxs.append(step_end_idx)

    for i in range(len(step_start)):
        if i == 0:
            # first step function
            stepped_profile[: step_begin_idxs[i]] = step_left[i]
            stepped_profile[step_begin_idxs[i] : step_end_idxs[i]] = step_middle[i]
            stepped_profile[step_end_idxs[i] :] = step_right[i]
        else:
            # first step function
            stepped_profile[step_end_idxs[i - 1] : step_begin_idxs[i]] = step_left[i]
            stepped_profile[step_begin_idxs[i] : step_end_idxs[i]] = step_middle[i]
            stepped_profile[step_end_idxs[i] :] = step_right[i]

    return stepped_profile


def plag_activity_calc(element, Xan, temp):
    """
    calculate the activity coefficient according to Dohment and Blundy (2014)
    This uses the equations 31 - 34 in that paper

    CURRENTLY THIS ONLY SUPPORTS THE +2 CATIONS: "Mg", "Ca", "Sr", "Ba", "Zn"

    Args:
        element (str): element for which the activity coefficient will be calculated
        Xan (scalar or array): molar fraction An (between 0 and 1)
        temp (scalar): temperature in degrees Celsius

    Returns:
        RTlngamma (array): RTln(activity coefficient) value in kJ/mol for the element specified
        gamma (array): activity coefficient value in J/mol for the element specified


    """

    T_K = temp + 273.15
    x_an_ci = 0.12 + 0.00038 * T_K

    if np.isscalar(Xan) is True:
        Xan = np.array(Xan)

    if element in ["Mg", "Ca", "Sr", "Ba", "Zn"]:
        for i in mendeleev.element("Ca").ionic_radii:
            if (i.coordination == "VIII") and (i.charge == 2):
                r_ref = i.ionic_radius / 100  # convert from pm to Angstroms

        for i in mendeleev.element(element).ionic_radii:
            if (i.coordination == "VIII") and (i.charge == 2):
                r_i = i.ionic_radius / 100  # convert from pm to Angstroms

        E_m = 120.03824
        # Eq 22 which is a modification of Eq 19 for the +2 cations
        # r_m = 1.2895 + 0.00013 * (T_K - 1563) + (-0.0952 + -0.00004 * (T_K - 1563)) * x_an

        # r_oab = 1.2895
        r_oab = 1.2778
        i_an = -(-4 - 0) * (1 - x_an_ci) ** 2
        i_an = np.full(Xan.shape, i_an)
        i_an[Xan > x_an_ci] = 0

        w_aban = np.full(Xan.shape, -4)
        w_aban[Xan > x_an_ci] = 0

        a2 = -0.0952
        b2 = -0.3686

        a = (
            -910.17
            * 8.314
            * E_m
            * (r_oab / 2 * (r_ref**2 - r_i**2) - 1 / 3 * (r_ref**3 - r_i**3))
            / 1000
            - w_aban
            - i_an
        )

        b = (
            -910.17
            * 8.314
            * (
                (E_m * a2 / 2 + r_oab / 2 * b2) * (r_ref**2 - r_i**2)
                - b2 / 3 * (r_ref**3 - r_i**3)
            )
            / 1000
            + 2 * w_aban
        )

        c = -w_aban - 910.17 * 8.314 * a2 * b2 / 2 * (r_ref**2 - r_i**2) / 1000

        RTlngamma = a + b * Xan + c * Xan**2

        return RTlngamma, np.exp(RTlngamma * 1000 / (8.314 * T_K))


def dohmen_activity_calc(element, Xan, temp, return_regression_stats=False):
    """
    calculate the activity coefficient according to Dohment and Blundy (2014)
    This uses the equations 31 - 34 in that paper

    Args:
        element (str): element for which the activity coefficient will be calculated
        Xan (scalar or array): molar fraction An (between 0 and 1)
        temp (scalar): temperature in degrees Celsius

    Returns:
        RTlngamma (scalar or array): RTln(activity coefficient) value in kJ/mol for the element specified
        gamma (scalar or array): activity coefficient value in J/mol for the element specified
        slope (scalar): slope of regression in RTln(activity coefficient) vs Xan space
        intercept (scalar): intercept of regression in RTln(activity coefficient) vs Xan space
    """
    RTlngamma, gamma = plag_activity_calc(element, Xan, temp)
    x = np.linspace(0, 1, 101)
    X = sm.add_constant(x)
    model = sm.OLS(plag_activity_calc(element, x, temp)[0], X)
    result = model.fit()
    intercept, slope = result.params

    if return_regression_stats is True:
        return RTlngamma, gamma, slope, intercept, result
    else:
        return RTlngamma, gamma, slope, intercept


def diffuse_forward_halfspace(
    initial_profile,
    observed_profile,
    timegrid,
    diffusivity_profile,
    an_profile,
    slope,
    distance_profile,
    temp,
    boundary="infinite observed",
    local_minima=True,
):
    """
        Function for running a forward diffusion model for either Sr or Mg in plagioclase
        based on the discretized solution to Eq. 7 from Costa et al., 2003

    Args:
        initial_profile (ndarray): starting point for the diffusion model

        observed_profile (ndarray): measured trace element profile in plagioclase

        timegrid (ndarray): time grid array. This is the output from plag_diff.get_tgrid()

        diffusivity_profile (ndarray): diffusion coefficient for each point in the
        profile. This is the output from plag_diff.plag_diffusivity()

        an_profile (ndarray): observed anorthite profile.

        slope (scalar): slope from regression in -RTln(gamma) vs An. Output # 3 from
        plag_diff.dohmen_activity_calc(). units are kJ/mol

        distance_profile (ndarray): distance profile array. It is
        observed_profile.shape[0] points long and spaced by the analytical resolution

        temp (scalar): temperature in Celsius

        boundary (str, optional): How to treat the most rimward boundary. Defaults to
        'infinite observed'. Infinite observed fixes the composition of the most rimward
        boundary and treats it in contact with an infinite reservoir. Other option is
        'open' that does not fix the most rimward composition and lets it diffuse like
        the rest of the interior points. This may be useful if your transect is not necessarily
        at the rim of the grain.

        local_minima (bool, optional): Whether or not to run until a minimum misfit is found
        between diffusion model and observed data. Defaults to True. If True, model runs for
        number of iterations that is timegrid.shape[0]. Only works if there are no local minima and
        therefore is only recommended for monte carlo situations where an individual diffusion
        model has already been run to figure this out. If False, model runs until the misfit
        for the current iteration is greater than the misfit for the previous iteration and then
        stops
    """
    R = 8.314  # J/molK
    # half spacing in D grid
    D_half = (diffusivity_profile[1:] + diffusivity_profile[:-1]) / 2
    D_half = np.insert(D_half, 0, diffusivity_profile[0])

    # slope = As[-1]
    u = np.zeros(initial_profile.shape[0])
    # u at previous iteration
    u_n = initial_profile.copy()
    # number of points in space grid
    nx = initial_profile.shape[0]
    # number of points in time grid
    nt = timegrid.shape[0]
    # time grid spacing
    dt = timegrid[1] - timegrid[0]
    # space grid spacing
    dx = distance_profile[1] - distance_profile[0]
    # Xan profile
    an = an_profile.copy()

    # A / RT
    theta = slope * 1000 / (R * (temp + 273.15))

    # container for diffusion model at each iteration
    curves = np.zeros((nt, nx))

    r = (diffusivity_profile * (timegrid[1] - timegrid[0])) / dx**2

    if np.any(r > 0.5):
        raise Exception(
            """
            YOU DO NOT HAVE NUMERICAL STABILITY EVERYWHERE IN YOUR PROFILE.
            Remember this is D * dt / dx^2 < 0.5 for every point in the profile. 
            You likely need to adjust your time grid to have a smaller dT.
            """
        )

    if local_minima == True:
        for n in tqdm(range(0, int(nt)), total=nt, unit="timestep"):
            # splitting up the numerical solution into Ci+1, Ci, Ci-1 terms
            cplus_term = u_n[2:nx] * (
                D_half[1 : nx - 1]
                - D_half[1 : nx - 1] * theta / 2 * (an[2:nx] - an[1 : nx - 1])
            )
            c_term = u_n[1 : nx - 1] * (
                D_half[0 : nx - 2]
                + D_half[1 : nx - 1]
                + D_half[1 : nx - 1] * theta / 2 * (an[2:nx] - an[1 : nx - 1])
                - D_half[0 : nx - 2] * theta / 2 * (an[1 : nx - 1] - an[0 : nx - 2])
            )
            cminus_term = u_n[0 : nx - 2] * (
                D_half[0 : nx - 2]
                + D_half[0 : nx - 2] * theta / 2 * (an[1 : nx - 1] - an[0 : nx - 2])
            )

            # for every point in the middle of the  grid
            u[1 : nx - 1] = u_n[1 : nx - 1] + dt / dx**2 * (
                cplus_term - c_term + cminus_term
            )

            # for the end point assume that the i-1 point is the same as i+1
            u[0] = u_n[0] + dt / dx**2 * (
                u_n[1] * (D_half[0] - D_half[0] * theta / 2 * (an[1] - an[0]))
                - u_n[0]
                * (
                    D_half[1]
                    + D_half[0]
                    + D_half[0] * theta / 2 * (an[1] - an[0])
                    - D_half[1] * theta / 2 * (an[0] - an[1])
                )
                + u_n[1] * (D_half[1] + D_half[1] * theta / 2 * (an[0] - an[1]))
            )

            if boundary == "infinite observed":
                # infinite reservoir assumption that the rim concentration does not change
                u[-1] = observed_profile[-1]
            elif boundary == "open":
                u[-1] = u_n[-1] + dt / dx**2 * (
                    u_n[-2] * (D_half[-1] - D_half[-1] * theta / 2 * (an[-2] - an[-1]))
                    - u_n[-1]
                    * (
                        D_half[-2]
                        + D_half[-1]
                        + D_half[-1] * theta / 2 * (an[-2] - an[-1])
                        - D_half[-1] * theta / 2 * (an[-2] - an[-1])
                    )
                    + u_n[-1]
                    * (D_half[-1] + D_half[-1] * theta / 2 * (an[-1] - an[-2]))
                )

            # record model curve
            curves[n, :] = u

            # the current iteration becomes the starting point for diffusion
            # in thet next iteration
            u_n[:] = u

        bf_time, chi2 = fit_model(observed_profile, curves)

    elif local_minima == False:
        chi2_p = 10000000
        chi2_c = 999999
        count = 0
        while chi2_c < chi2_p:
            count += 1
            chi2_p = chi2_c
            # splitting up the numerical solution into Ci+1, Ci, Ci-1 terms
            cplus_term = u_n[2:nx] * (
                D_half[1 : nx - 1]
                - D_half[1 : nx - 1] * theta / 2 * (an[2:nx] - an[1 : nx - 1])
            )
            c_term = u_n[1 : nx - 1] * (
                D_half[0 : nx - 2]
                + D_half[1 : nx - 1]
                + D_half[1 : nx - 1] * theta / 2 * (an[2:nx] - an[1 : nx - 1])
                - D_half[0 : nx - 2] * theta / 2 * (an[1 : nx - 1] - an[0 : nx - 2])
            )
            cminus_term = u_n[0 : nx - 2] * (
                D_half[0 : nx - 2]
                + D_half[0 : nx - 2] * theta / 2 * (an[1 : nx - 1] - an[0 : nx - 2])
            )

            # for every point in the middle of the  grid
            u[1 : nx - 1] = u_n[1 : nx - 1] + dt / dx**2 * (
                cplus_term - c_term + cminus_term
            )

            # for the end point assume that the i-1 point is the same as i+1
            u[0] = u_n[0] + dt / dx**2 * (
                u_n[1] * (D_half[0] - D_half[0] * theta / 2 * (an[1] - an[0]))
                - u_n[0]
                * (
                    D_half[1]
                    + D_half[0]
                    + D_half[0] * theta / 2 * (an[1] - an[0])
                    - D_half[1] * theta / 2 * (an[0] - an[1])
                )
                + u_n[1] * (D_half[1] + D_half[1] * theta / 2 * (an[0] - an[1]))
            )

            if boundary == "infinite observed":
                # infinite reservoir assumption that the rim concentration does not change
                u[-1] = observed_profile[-1]
            elif boundary == "open":
                u[-1] = u_n[-1] + dt / dx**2 * (
                    u_n[-2] * (D_half[-1] - D_half[-1] * theta / 2 * (an[-2] - an[-1]))
                    - u_n[-1]
                    * (
                        D_half[-2]
                        + D_half[-1]
                        + D_half[-1] * theta / 2 * (an[-2] - an[-1])
                        - D_half[-1] * theta / 2 * (an[-2] - an[-1])
                    )
                    + u_n[-1]
                    * (D_half[-1] + D_half[-1] * theta / 2 * (an[-1] - an[-2]))
                )

            # record model curve
            curves[count, :] = u

            # the current iteration becomes the starting point for diffusion
            # in thet next iteration
            u_n[:] = u

            chi2_c = np.sum((u - observed_profile) ** 2 / observed_profile)

            if count == nt:
                break

        bf_time, chi2 = fit_model(observed_profile, curves[:count, :])

    return curves, bf_time, chi2
