# Copyright (c) 2018, Hyeokhyen Kwon
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import numpy as np
from scipy.interpolate import interp1d
from Hammerla_et_al import ecdfRep

def structECDF(data, # shape: sensor channel x window dimension
    is_intrp=True,
    # multi-scale
    is_ms=True,
    multiScale=[.25,.5,1.], # 1/4 | 1/2 | original scales
    # multi-window
    is_mw=True,  
    multiWindow=[.25,.5,1.], # 1/4 | 1/2 | original window sizes
    # convolution
    is_cv=True, 
    convSubWinsize=0.5, convSubWinSlide=0.5,
    # ECDF representation
    n_ecdf_coeff=10):
    #
    #   Estimate structural ecdf-representation according to 
    #
    #   Kwon, H., Abowd, G. D., & Ploetz, T. (2018, October). 
    #   Adding structural characteristics to distribution-based accelerometer
    #   representations for activity recognition using wearables. 
    #   In Proceedings of the 2018 ACM International Symposium on Wearable Computers
    #   (pp. 72-75). ACM.
    #
    #   Hyeok Kwon '18
    #   hyeokhyen@gatech.edu
    #
    if is_ms:
        data = f_multiscale(data, multiScale, is_intrp)
    if is_mw:
        data = f_multiwindow(data, multiWindow, is_intrp)
    if is_cv:
        data = f_convolution(data, 
            convSubWinsize, convSubWinSlide, is_intrp)
    # refer Hammerla et al. for ECDF_representation code
    feature = ecdfRep(data.T, n_ecdf_coeff)
    return feature

def f_multiscale(data, multiScale, is_intrp):
    ch, dim = data.shape
    output = np.empty((len(multiScale)*ch, dim))
    output[:] = np.nan
    for i, ms in enumerate(multiScale):
        idx_samples = np.int32(np.around(
            np.linspace(0, dim-1, num=int(dim*ms))))     
        if is_intrp:
            f_interp1d = interp1d(idx_samples, data[:,idx_samples], axis=1)
            output[i*ch:(i+1)*ch,:] = f_interp1d(np.arange(dim))
        else:
            output[i*ch:(i+1)*ch,idx_samples] = data[:,idx_samples]
    return output

def f_multiwindow(data, multiWindow, is_intrp):
    ch, dim = data.shape
    output = np.empty((len(multiWindow)*ch, dim))
    output[:] = np.nan
    for i, mw in enumerate(multiWindow):
        _dim_mw = int(dim*mw)
        if is_intrp:
            idx_pts = np.int32(np.around(
                        np.linspace(0, dim-1, num=_dim_mw)))
            f_interp1d = interp1d(idx_pts, data[:,-_dim_mw:], axis=1)
            output[i*ch:(i+1)*ch,:] = f_interp1d(np.arange(dim))
        else:
            output[i*ch:(i+1)*ch,-_dim_mw:] = data[:,-_dim_mw:]
    return output

def f_convolution(data, convSubWinsize, convSubWinSlide, is_intrp):
    ch, dim = data.shape
    subWinSize = int(np.ceil(dim*convSubWinsize))
    slideSize = int(np.ceil(subWinSize*convSubWinSlide))
    n_subWin = int(np.ceil((dim-subWinSize)/slideSize)+1)
    
    idx_pts = np.int32(np.around(
        np.linspace(0, dim-1, num=subWinSize)))
    output = np.empty((n_subWin, ch, dim))
    output[:] = np.nan
    for f in range(n_subWin):
        window_remain = dim - f*slideSize
        if window_remain == 0:
            break
        if window_remain > subWinSize:
            subframe = data[:,f*slideSize:f*slideSize+subWinSize]
        else:
            subframe = data[:, -subWinSize:]

        if is_intrp:
            f_interp1d = interp1d(idx_pts, subframe, axis=1)
            output[f,:,:] = f_interp1d(np.arange(dim))
        else:
            output[f,:,idx_pts] = subframe

    output = output.reshape(np.prod(output.shape[:-1]), output.shape[-1])
    return output
