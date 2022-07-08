import time

time0 = time.time()
time00 = time.time()

"""
Datafile
"""
DATA = "client-feature-usage.xls"

"""
Optimisation variables
"""
#0 = Optimisation by client growth, 1 = Optimisation by client revenue:
SCHEMES = [0]
# Optimisation is performed using segmentation of subscription_quantity:
OPTIMISATION_SEGMENTS = [
    [2, 10000],
    [25,10000],
    [40,10000]
    ]
# Number of roadmap items included in the analysis; the least relevant apps are omitted
# Warning: Heavily influences required memory usage and computation time
ROADMAP_LENGTH = 6
# Ignore already released roadmap items:
ROADMAP_START = [
    'App.D',
    'App.G'
]
"""
Post-optimisation analysis
"""
# For visualising and analysing the results
# SEGMENTS = [
#     [0,1],
#     [2,9],
#     [10,24],
#     [25,49],
#     [50,10000]
# ]
SEGMENTS = [
    [0, 1],
    [2, 24],
    [25, 39],
    [40, 10000]
]