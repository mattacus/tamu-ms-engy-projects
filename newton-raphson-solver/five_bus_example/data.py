# Five-bus system data from Tables 1, 2, and 3
#
# Bus data format: [bus_number, type, P_G, Q_G, P_L, Q_L, V_spec, theta_spec]
# where:
#   bus_number: Bus identification number (1-5)
#   type: 0=slack/swing, 1=PV/voltage-controlled, 2=PQ/load
#   P_G: Real power generation in per unit
#   Q_G: Reactive power generation in per unit (for PQ buses, this is unspecified)
#   P_L: Real power load in per unit
#   Q_L: Reactive power load in per unit
#   V_spec: Specified voltage magnitude in per unit (for slack and PV buses)
#   theta_spec: Specified voltage angle in radians

buses = [
    [1, 0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    [2, 2, 0.0, 0.0, 8.0, 2.8, 0.0, 0.0],
    [3, 1, 5.2, 0.0, 0.8, 0.4, 1.05, 0.0],
    [4, 2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [5, 2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
]

# Line/Transformer data format: [from_bus, to_bus, R, X, B_half, type]
# where:
#   from_bus, to_bus: Connected bus numbers
#   R: Series resistance in per unit
#   X: Series reactance in per unit
#   B_half: Half of total shunt susceptance (B'/2) in per unit
#   type: 'line' or 'transformer'

lines = [
    # Transformers (from Table 3)
    [1, 5, 0.00150, 0.02, 0.0, 'transformer'],
    [3, 4, 0.00075, 0.01, 0.0, 'transformer'],

    # Transmission Lines (from Table 2)
    [2, 4, 0.0090, 0.100, 0.86, 'line'],
    [2, 5, 0.0045, 0.050, 0.44, 'line'],
    [4, 5, 0.00225, 0.025, 0.22, 'line']
]

# Additional system parameters
system_base_mva = 100  # System base MVA (assumed)

# Bus limits (from Table 1)
bus_limits = {
    # Bus 3 generator limits
    3: {'Q_max': 4.0, 'Q_min': -2.8}  # Q limits in per unit
}
