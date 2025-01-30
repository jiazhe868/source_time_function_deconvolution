import numpy as np
import obspy
import sys
from obspy import read
import cvxpy as cp
import matplotlib.pyplot as plt
from scipy.signal import convolve, tukey
from scipy.fftpack import fft, ifft
from scipy.linalg import toeplitz, solve, lstsq
from scipy.optimize import minimize, nnls

def gaussian_time_series(dt, t_begin, t_end, mu, dura, amplitude):
    t = np.arange(t_begin, t_end, dt)
    y = amplitude * np.exp(-((t - mu) ** 2) / (2 * (dura/5) ** 2))
    return t, y

def decon_optim(w, u, lambda_reg, d2_reg, group_size=20):
    """
    Performs time-domain deconvolution with non-negative constraints and mixed L1-L2 regularization,
    enforcing x[0] = 0 and x[-1] = 0.
    Parameters:
    w (numpy.ndarray): Main time series (observed data), a 1D array of length N.
    u (numpy.ndarray): Kernel time series (impulse response), a 1D array of length N.
    lambda_reg (float): Regularization parameter.
    alpha (float): Mixing parameter between L1 and L2 regularization (0 <= alpha <= 1).
    """
    # Ensure that w and u are 1D numpy arrays of the same length
    w = np.asarray(w).flatten()
    u = np.asarray(u).flatten()
    N = len(w)
    if len(u) != N:
        raise ValueError("The time series w and u must have the same length.")

    # Construct the convolution matrix U
    U = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i - j >= 0:
                U[i, j] = u[i - j]

    # Define the optimization variable
    x = cp.Variable(N, nonneg=True)

    # Enforce x[0] = 0 and x[N-1] = 0
    constraints = [x[0] == 0, x[N-1] == 0]
    D2 = cp.diff(x, k=2)
    groups = []
    for start in range(1, N - 1, group_size):
        end = min(start + group_size, N - 1)
        groups.append(x[start:end])

    # Compute the L12 norm over the groups
    l12_norm = cp.sum([cp.norm2(group) for group in groups])

    # Define the objective function
    residual = U @ x - w

    # Compute scaling factor for lambda_reg
    scaling_factor = np.linalg.norm(U.T @ w)
    lambda_reg_scaled = lambda_reg * scaling_factor
    d2_reg_scaled = d2_reg * scaling_factor

    # Objective function with mixed L1-L2 regularization
    objective = cp.Minimize(
        cp.sum_squares(residual) + cp.sum_squares(D2) * d2_reg_scaled +
        l12_norm * lambda_reg_scaled
#        alpha * lambda_reg_scaled * cp.norm1(x) +
#        (1 - alpha) * lambda_reg_scaled * cp.sum_squares(x) 
    )

    # Define the problem
    problem = cp.Problem(objective, constraints)

    # Solve the problem
    problem.solve(solver=cp.SCS, verbose=False)

    # Get the solution
    x_value = x.value

    return x_value


def apply_taper(data, win_b, win_e, delta, alpha):
    """
    Apply a taper to the left side of the data (first `win_b` seconds) to smoothly reduce the amplitude to 0.
    
    Parameters:
    - data: The waveform data to taper
    - win_b: The left window length (e.g., -5s) to taper
    - win_e: The right window length (e.g., 30s) to remain unaffected
    - delta: Sampling interval of the data
    - alpha: The proportion of the Tukey window used for tapering
    
    Returns:
    - Tapered data
    """
    npts_b = int(win_b / delta)  # Number of points for the left taper
    npts_e = int(win_e / delta)  # Total number of points for the remaining data

    # Create a Tukey window for tapering only the left part of the data
    taper = tukey(npts_e, alpha=alpha)
    
    # Apply taper to the left part of the data (smooth to 0 for win_b)
    tapered_data = data.copy()
    tapered_data[:npts_b] *= taper[:npts_b]  # Apply taper to the left window
    
    return tapered_data

def source_time_function_deconvolution(sac_file1, sac_file2, iterations=100, step_size=0.1):
    # Read the SAC files
    st1 = read(sac_file1)
    st2 = read(sac_file2)

    # Extract data and metadata
    tr1 = st1[0]  # Complex waveform
    tr2 = st2[0]  # Empirical Green's function

    # Extract sampling rate from SAC headers
    delta1 = tr1.stats.delta  # Time step (sampling interval) of sac_file1
    delta2 = tr2.stats.delta  # Time step (sampling interval) of sac_file2

    print(f"Sampling rate 1: {1/delta1}, Sampling rate 2: {1/delta2}")
    delta=delta1
    # If sampling rates are not the same, decimate the higher one
    if delta1 != delta2:
        if delta1 > delta2:
            factor = int(delta1 / delta2)
            tr1.data = decimate(tr1.data, factor)
            tr1.stats.delta = delta2
            delta=delta2
            print(f"Decimated {sac_file1} to match sampling rate of {sac_file2}")
        else:
            factor = int(delta2 / delta1)
            tr2.data = decimate(tr2.data, factor)
            tr2.stats.delta = delta1
            delta=delta1
            print(f"Decimated {sac_file2} to match sampling rate of {sac_file1}")

    # Check if t1 marker exists
    if not hasattr(tr1.stats.sac, 't1'):
        print(f"Error: SAC file {sac_file1} does not have a t1 marker.")
        return

    # Extract begin time from SAC headers (relative to the event origin)
    begin_time1 = tr1.stats.sac.b
    begin_time2 = tr2.stats.sac.b

    # Get the t1 marker from SAC headers
    t1_marker = tr1.stats.sac.t1
    t1_1 = tr1.stats.sac.t1  # Time of t1 marker in seconds relative to trace start
    t1_2 = tr2.stats.sac.t1

    # Define the time window: -5s before t1 to 30s after t1
    win_b = -10  # Time window start (5 seconds before t1)
    win_e = 50  # Time window end (30 seconds after t1)
    
    # Calculate the relative start and end time with respect to begin time (b)
    start_time1 = t1_1 + win_b - begin_time1
    end_time1 = t1_1 + win_e - begin_time1

    start_time2 = t1_2 + win_b - begin_time2
    end_time2 = t1_2 + win_e - begin_time2

    # Slice both traces to the desired time window
    tr1_windowed = tr1.slice(starttime=tr1.stats.starttime + start_time1, 
                             endtime=tr1.stats.starttime + end_time1)
    tr2_windowed = tr2.slice(starttime=tr2.stats.starttime + start_time2, 
                             endtime=tr2.stats.starttime + end_time2)

    # Sharp taper on the left side of tr1_windowed data
    taper_length_samples = int(win_b / delta1)  # Convert the taper length to samples
    tr1_windowed.taper(max_percentage= -win_b / (-win_b + win_e), type='cosine')
    tr2_windowed.taper(max_percentage= -win_b / (-win_b + win_e), type='cosine')
    # Extract data for iterative deconvolution
    data1 = tr1_windowed.data
    data2 = tr2_windowed.data
    data1_normalized = data1 / np.max(np.abs(data1))
    data2_normalized = data2 / np.max(np.abs(data2))
    deconvolved_stf = decon_optim(data1_normalized, data2_normalized, 0.05, 0)

    # Create synthetic seismogram by convolving the Green's function with the retrieved STF
    synthetic_seismogram = convolve(deconvolved_stf, data2_normalized, mode='full')[:len(data1_normalized)]  # Trim to match length of data1

    # Normalize the data (complex waveform, synthetic seismogram, and STF)
    synthetic_seismogram_normalized = synthetic_seismogram / np.max(np.abs(synthetic_seismogram))
    deconvolved_stf_normalized = deconvolved_stf / np.max(np.abs(deconvolved_stf))


    # Create time axis relative to t1 marker for plotting
    npts = len(deconvolved_stf)
    time_axis = np.linspace(win_b, win_e, npts)
    
    # Save the deconvolved source time function as a new SAC file
    stf_trace = tr1.copy()  # Use the original trace as a template
    stf_trace.data = deconvolved_stf
    output_filename = f"stf_{sac_file1.split('.')[1]}.z"
    stf_trace.write(output_filename, format='SAC')

    print(f"Deconvolution complete.")

    # Plot the input waveforms and the deconvolved source time function
    plt.figure(figsize=(10, 6))
    # Plot complex waveform (input 1) and synthetic seismogram (in red)
    plt.subplot(3, 1, 1)
    plt.plot(np.linspace(win_b, win_e, len(data1_normalized)), data1_normalized, label=f"obs_{sac_file1.split('.')[1]}")
    plt.plot(np.linspace(win_b, win_e, len(synthetic_seismogram_normalized)), synthetic_seismogram_normalized, label='Syn', color='red')
    plt.axvline(x=0, color='k', linestyle='--', label='t1')
    plt.title(f'OBS_{sac_file1.split(".")[1]}')
    plt.ylabel('Amplitude (Normalized)')
    plt.legend(loc='lower left')
    
    # Plot empirical Green's function (input 2)
    plt.subplot(3, 1, 2)
    plt.plot(np.linspace(win_b, win_e, len(data2_normalized)), data2_normalized, label=f"syn_{sac_file1.split('.')[1]}")
    plt.axvline(x=0, color='k', linestyle='--', label='t1')
    plt.title(f'EGF_{sac_file1.split(".")[1]}')
    plt.ylabel('Amplitude (Normalized)')
    plt.legend(loc='lower left')  # Align legend to the left
    
    # Plot deconvolved source time function (output)
    x_shifted = np.zeros_like(deconvolved_stf_normalized)
    x_shifted[-int(win_b/delta):] = deconvolved_stf_normalized[:len(deconvolved_stf_normalized) + int(win_b/delta)]

    plt.subplot(3, 1, 3)
    plt.plot(time_axis, x_shifted, label=f'STF')
    plt.axvline(x=0, color='k', linestyle='--', label='t1')
    plt.title(f'STF_{sac_file1.split(".")[1]}')
    plt.xlabel('Time (s) relative to t1 marker')
    plt.ylabel('Amplitude (Normalized)')
    plt.legend(loc='lower left')

    plt.tight_layout()

    # Save the plot as a PDF
    output_pdf = f'stf_{sac_file1.split(".")[1]}.pdf'
    plt.savefig(output_pdf)
    plt.show()

# Ensure correct number of arguments
if len(sys.argv) != 3:
    print("Usage: python decon4.py <complex_waveform_sac> <empirical_greens_function_sac>")
    sys.exit(1)

# Get input file names from command line arguments
sac_file1 = sys.argv[1]
sac_file2 = sys.argv[2]

# Run the deconvolution
source_time_function_deconvolution(sac_file1, sac_file2)

