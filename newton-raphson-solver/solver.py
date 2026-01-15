"""
================================================================================
 File:        solver.py
 Author:      Matt Lee
 Course:      ECEN 615
 Date:        9-30-2025

 Description: Python power flow solver for Homework 2, problem 1.  For this project,
 the Github Copilot tool was used to assist in documentation, formatting, language features
 (such as working with complex numbers), and some implementation suggestions.

 To run:
 1. Ensure you have Python 3.x installed.
 2. Install the required packages: numpy, pandas
 3. Run the script using the command: python solver.py five_bus_example/ for the five bus system

================================================================================
"""

import numpy as np
import pandas as pd


def construct_ybus(lines, n):
    """
    Construct the admittance matrix Y-bus from line data
    Assumes all G (conductance) values are zero
    """

    # Initialize Y-bus matrix
    Y = np.zeros((n, n), dtype=complex)

    for line in lines:
        # Convert bus numbers to be 0-based
        from_bus = line[0] - 1
        to_bus = line[1] - 1
        R = line[2]
        X = line[3]
        B_half = line[4]

        # Calculate the series admittance (Y = 1/(R + jX)), but assume all R = 0
        Z = complex(0, X)
        Y_series = 1 / Z if Z != 0 else 0

        # Update diagonal elements (with shunt admittances)
        Y[from_bus, from_bus] += Y_series + 1j * B_half
        Y[to_bus, to_bus] += Y_series + 1j * B_half

        # Update off-diagonal elements
        Y[from_bus, to_bus] -= Y_series
        Y[to_bus, from_bus] -= Y_series

    return Y


def calc_power_injections(Y, V):
    """
    Calculate power injections P and Q at all buses from the Y bus and voltage vector
    (This is part of f(x) in Newton-Raphson method)
    """

    I = Y @ V  # I_i = sum(j=1 to n) Y_ij × V_j
    S = V * np.conj(I)  # S_i = V_i × I_i*
    # using real and imaginary rather than trig functions since it is more efficient
    P = S.real
    Q = S.imag
    return P, Q


def calc_jacobian(Y, V, P_calc, Q_calc, p_eq_buses, q_eq_buses, theta_buses, vMag_buses):
    """
    Calculate the Jacobian matrix for Newton-Raphson power flow.
    Assumes G=0 (all resistances negligible) for simplified calculations.
    """

    V_mag = np.abs(V)
    theta = np.angle(V)

    n_eq = len(p_eq_buses) + len(q_eq_buses)
    n_var = len(theta_buses) + len(vMag_buses)

    J = np.zeros((n_eq, n_var))

    # Calculate Jacobian submatrices (blocks)

    # Top left (J_11) block (∂P_i/∂θ)
    for row, i in enumerate(p_eq_buses):
        for col, k in enumerate(theta_buses):
            if i == k:
                # Diagonal element: ∂P_i/∂θ_i = -Q_i - |V_i|² × B_ii
                J[row, col] = -Q_calc[i] - V_mag[i]**2 * Y[i, i].imag
            else:
                # Off-diagonal element: ∂P_i/∂θ_k = |V_i||V_k| × [G_ik×sin(θ_i - θ_k) - B_ik × cos(θ_i - θ_k)] where G = 0
                J[row, col] = V_mag[i] * V_mag[k] * \
                    (-Y[i, k].imag * np.cos(theta[i] - theta[k]))

    # Top right (J_12) block (∂P_i/∂V)
    for row, i in enumerate(p_eq_buses):
        for col, k in enumerate(vMag_buses):
            if i == k:
                # Diagonal element: ∂P_i/∂V_i = (P_i/|V_i|) + |V_i| × G_ii where G = 0
                J[row, len(theta_buses) + col] = (P_calc[i] / V_mag[i])
            else:
                # Off-diagonal element: ∂P_i/∂V_k = V_i × [G_ik×cos(θ_i - θ_k) + B_ik×sin(θ_i - θ_k)] where G = 0
                J[row, len(theta_buses) + col] = V_mag[i] * \
                    (Y[i, k].imag * np.sin(theta[i] - theta[k]))

    # Bottom left (J_21) block (∂Q_i/∂θ)
    for row, i in enumerate(q_eq_buses):
        for col, k in enumerate(theta_buses):
            if i == k:
                # Diagonal element: ∂Q_i/∂θ_i = P_i - V_i² × G_ii where G = 0
                J[len(p_eq_buses) + row, col] = (P_calc[i])
            else:
                # Off-diagonal element: ∂Q_i/∂θ_k = -V_i×V_k × [G_ik×cos(θ_i - θ_k) + B_ik×sin(θ_i - θ_k)] where G = 0
                J[len(p_eq_buses) + row, col] = -V_mag[i] * V_mag[k] * \
                    (Y[i, k].imag * np.sin(theta[i] - theta[k]))

    # Bottom right (J_22) block (∂Q/∂V)
    for row, i in enumerate(q_eq_buses):
        for col, k in enumerate(vMag_buses):
            if i == k:
                # Diagonal element: ∂Q_i/∂V_i = (Q_i/V_i) - V_i × B_ii
                J[len(p_eq_buses) + row, len(theta_buses) + col] = (
                    Q_calc[i] / V_mag[i] - V_mag[i] * Y[i, i].imag
                )
            else:
                # Off-diagonal element: ∂Q_i/∂V_k = V_i × [G_ik×sin(θ_i - θ_k) - B_ik×cos(θ_i - θ_k)] where G = 0
                J[len(p_eq_buses) + row, len(theta_buses) + col] = V_mag[i] * (
                    -Y[i, k].imag * np.cos(theta[i] - theta[k])
                )

    return J


def lu_decomposition(A, b):
    """
    Solve the linear system Ax = LUx = b using the LU decomposition algorithm from Lecture 6 with Gaussian elimination.
    Uses the method with 1's on the diagonal of L matrix, no pivoting
    """
    n = len(A)
    A = A.copy()
    b = b.copy()

    # LU Decomposition
    for i in range(1, n):
        for j in range(i):
            A[i, j] = A[i, j] / A[j, j]
            for k in range(j + 1, n):
                A[i, k] = A[i, k] - A[i, j] * A[j, k]

    # print the L and U matrices (debug)
    L = np.tril(A, -1) + np.eye(n)
    U = np.triu(A)
    print("L matrix from LU decomposition:\n", pd.DataFrame(L).round(4))
    print("U matrix from LU decomposition:\n", pd.DataFrame(U).round(4))

    # Forward substitution (solves b = Ly)
    y = np.zeros(n)
    for i in range(n):
        y[i] = b[i]
        for j in range(i):
            y[i] = y[i] - A[i, j] * y[j]

    # Backward substitution (solves y = Ux)
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = y[i]
        for j in range(i + 1, n):
            x[i] = x[i] - A[i, j] * x[j]
        x[i] /= A[i, i]  # The A[i,i] values are != 0 if it is nonsingular

    return x


def solve_power_flow(buses, lines, base_mva=100, tolerance_mva=0.1, max_iterations=20):
    """
    Solve power flow using the iterative Newton-Raphson method.
    """

    # initialize parameter values
    n_buses = len(buses)
    tolerance_pu = tolerance_mva / base_mva  # convergence tolerance in per unit

    # Step 1: Build the Y-bus matrix
    Y = construct_ybus(lines, len(buses))
    print(f"Y-bus Matrix:\n{pd.DataFrame(Y).round(4)}")

    # Step 2: Initialize starting parameters
    # Initialize voltages and power injection
    V = np.ones(n_buses, dtype=complex)
    P = np.zeros(n_buses)  # P specified
    Q = np.zeros(n_buses)  # Q specified

    for i, bus in enumerate(buses):
        bus_type = bus[1]
        if bus_type == 0:  # Slack bus
            V[i] = bus[6] * np.exp(1j * bus[7])  # initial guess
        elif bus_type == 1:  # PV bus
            V[i] = bus[6] * np.exp(1j * 0)
        # Calculate initial power injections
        P[i] = bus[2] - bus[4]  # P_G - P_L
        Q[i] = bus[3] - bus[5]  # Q_G - Q_L

    print(
        f"Initial Voltages and Power Injections:\n{pd.DataFrame({'V': V, 'P': P, 'Q': Q}).round(4)}")

    # Step 3: Iterative Newton-Raphson solution
    for iteration in range(max_iterations):
        # Calculate current power injections at all buses
        P_calc, Q_calc = calc_power_injections(Y, V)

        # Form f(x) vector (power flow equations for buses with power equations) for each node
        pv_buses = [i for i, bus in enumerate(buses) if bus[1] == 1]
        pq_buses = [i for i, bus in enumerate(buses) if bus[1] == 2]
        p_eq_buses = sorted(pv_buses + pq_buses)
        q_eq_buses = sorted(pq_buses)
        theta_buses = sorted(pv_buses + pq_buses)
        vMag_buses = sorted(pq_buses)

        # print all equation buses
        print(f"  P equation buses: {[i+1 for i in p_eq_buses]}")
        print(f"  Q equation buses: {[i+1 for i in q_eq_buses]}")
        print(f"  Theta variable buses: {[i+1 for i in theta_buses]}")
        print(
            f"  Voltage magnitude variable buses: {[i+1 for i in vMag_buses]}")

        # f(x) = calculated power - specified power (this is what should be zero at solution)
        # delta_P_i = P_calc_i - P_spec_i
        delta_P_i = P_calc[p_eq_buses] - P[p_eq_buses]
        # delta_Q_i = Q_calc_i - Q_spec_i
        delta_Q_i = Q_calc[q_eq_buses] - Q[q_eq_buses]
        f_x = np.concatenate([delta_P_i, delta_Q_i])

        # Check convergence using infinity norm of f(x)
        f_x_norm = np.linalg.norm(f_x, ord=np.inf)
        max_mismatch_mva = f_x_norm * base_mva

        # Store voltage magnitude and angle for current iteration
        V_mag = np.abs(V)
        V_angle = np.angle(V)

        # Print iteration results
        print(f"Iteration {iteration}:")
        print((f" Current mismatch f(x) : {(f_x * base_mva).round(6)} MVA"))
        print(f"  ||f(x)||: {f_x_norm:.6f} pu ({max_mismatch_mva:.6f} MVA)")
        print("  Bus voltages:")
        for i in range(n_buses):
            bus_type = ['Slack', 'PV', 'PQ'][buses[i][1]]
            print(
                f"    Bus {i+1} ({bus_type}): {V_mag[i]:.6f} pu ∠ {np.degrees(V_angle[i]):.3f}°")

        if max_mismatch_mva < tolerance_mva:
            print(f"\nConverged in {iteration} iterations!")
            print(f"Final ||f(x)|| = {f_x_norm:.8f} pu")

            # Calculate final generator outputs
            P_final, Q_final = calc_power_injections(Y, V)

            # Generator reactive power output and slack bus real power output
            Q_gen = np.zeros(n_buses)
            P_slack = np.zeros(n_buses)

            print("\nFinal Results:")

            for i in range(n_buses):
                bus_type = ['Slack', 'PV', 'PQ'][buses[i][1]]
                if buses[i][1] in [0, 1]:  # Get reactive power for PV and slack buses
                    Q_gen[i] = Q_final[i] + buses[i][5]
                    print(
                        f"  Bus {i+1} ({bus_type}) reactive power: {Q_gen[i]:.4f} pu ({Q_gen[i]*base_mva:.1f} MVAr)")
                if buses[i][1] == 0:  # Get real power for slack bus
                    P_slack[i] = P_final[i] + buses[i][4]
                    print(
                        f"  Bus {i+1} ({bus_type}) real power: {P_slack[i]:.4f} pu ({P_slack[i]*base_mva:.1f} MW)")

            break

        # Update step: v_next = v_current - J(v_current)^-1 * f(x_current)

        # Calculate Jacobian matrix
        J = calc_jacobian(Y, V, P_calc, Q_calc, p_eq_buses,
                          q_eq_buses, theta_buses, vMag_buses)
        # Create labels for rows and columns
        p_eq_labels = [f"P_{i+1}" for i in (p_eq_buses)]
        q_eq_labels = [f"Q_{i+1}" for i in (q_eq_buses)]
        theta_labels = [f"θ_{i+1}" for i in (theta_buses)]
        vMag_labels = [f"|V|_{i+1}" for i in (vMag_buses)]

        row_labels = p_eq_labels + q_eq_labels
        col_labels = theta_labels + vMag_labels

        print(
            f"  Jacobian Matrix:\n{pd.DataFrame(J, index=row_labels, columns=col_labels).round(4)}")

        # Solve using LU decomposition: J * dx = -f(x)
        # we solve for -f(x) because we want f(x + dx) ≈ 0
        dx = lu_decomposition(J, -f_x)
        # Update angles for PV and PQ buses
        for idx, bus_idx in enumerate(theta_buses):
            V_angle[bus_idx] += dx[idx]
        # Update magnitudes for PQ buses
        for idx, bus_idx in enumerate(vMag_buses):
            V_mag[bus_idx] += dx[len(theta_buses) + idx]

        # Reconstruct voltage vector for next iteration (v_next)
        V = V_mag * np.exp(1j * V_angle)


def print_data(buses, lines, bus_limits, base_mva):
    """
    Print the starting bus and line data in a human readable format
    """

    print("\nBus Data:")
    print("Bus\tType\tP_G\tQ_G\tP_L\tQ_L\tV_spec\tTheta_spec")
    for bus in buses:
        print("\t".join(f"{x}" for x in bus))

    print("\nLine Data:")
    print("From\tTo\tR\tX\tB/2\tType")
    for line in lines:
        print("\t".join(f"{x}" for x in line))

    print(f"\nSystem Base MVA: {base_mva}")
    print("\nBus Limits:")
    for bus_num, limits in bus_limits.items():
        print(
            f"Bus {bus_num}: Q_min = {limits['Q_min']} pu, Q_max = {limits['Q_max']} pu")
    print("\n")


# Example usage
if __name__ == "__main__":
    # Import the bus data from specified folder
    import sys
    import os

    # Get folder name from command line or user input
    if len(sys.argv) > 1:
        folder_name = sys.argv[1]
    else:
        folder_name = input("Enter the folder name containing data.py: ")

    # Construct the path to the data folder
    data_folder_path = os.path.join(os.path.dirname(__file__), folder_name)

    # Check if the folder exists
    if not os.path.exists(data_folder_path):
        raise ImportError(
            f"Folder '{folder_name}' not found in {os.path.dirname(__file__)}")

    # Check if data.py exists in the folder
    data_file_path = os.path.join(data_folder_path, 'data.py')
    if not os.path.exists(data_file_path):
        raise ImportError(f"data.py file not found in folder '{folder_name}'")

    # Add the folder to the Python path
    sys.path.append(data_folder_path)

    try:
        from data import buses, lines, bus_limits, system_base_mva
        print_data(buses, lines, bus_limits, system_base_mva)
        print(f"Running power flow with data from '{folder_name}' folder...")

        # Pass parameters to solve the power flow (we don't have to code generator reactive power limits)
        results = solve_power_flow(
            buses, lines, base_mva=system_base_mva)
    except ImportError as e:
        raise ImportError(
            f"Could not import buses and lines from data.py in '{folder_name}': {e}")
