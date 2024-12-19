# Shaft# class post-processing
shaft_rules = [
    {   "conditions": [
            lambda x: "NO1" in x,
            lambda x: "SHAFT" in x
        ],
        "action": 'Shaft1'
    },
    {   "conditions": [
            lambda x: "NO1" in x,
            lambda x: "Shaft" in x
        ],
        "action": 'Shaft1'
    },
    {   "conditions": [
            lambda x: "NO2" in x,
            lambda x: "Shaft" in x
        ],
        "action": 'Shaft2'
    },
    {   "conditions": [
            lambda x: "NO2" in x,
            lambda x: "SHAFT" in x
        ],
        "action": 'Shaft2'
    },

    {   "conditions": [
            lambda x: "NO1" not in x,
            lambda x: "NO2" not in x,
            lambda x: "SHAFT" not in x
        ],
        "action": 'Shaft1'
    },

    {   "conditions": [
            lambda x: "NO1" not in x,
            lambda x: "NO2" not in x,
            lambda x: "SHAFT" in x,
            lambda x: "(P)" in x
        ],
        "action": 'Shaft2'
    },

    {   "conditions": [
            lambda x: "NO1" not in x,
            lambda x: "NO2" not in x,
            lambda x: "SHAFT" in x,
            lambda x: "(S)" in x
        ],
        "action": 'Shaft1'
    },

    {   "conditions": [
            lambda x: "NO1" not in x,
            lambda x: "NO2" not in x,
            lambda x: "SHAFT" in x,
            lambda x: "(S)" not in x,
            lambda x: "(P)" not in x
        ],
        "action": 'Shaft1'
    },
]


# ME# class post-processing
ME_rules = [
    {   "conditions": [
            lambda x: "ME" in x,
            lambda x: "(P)" not in x,
            lambda x: "(S)" not in x,
            lambda x: "GE" not in x,
            lambda x: "FLOW" in x,
        ],
        "action": 'ME1Flow'
    },
    {   "conditions": [
            lambda x: "ME" in x,
            lambda x: "(P)" in x,
            lambda x: "FLOW" in x,
        ],
        "action": 'ME2Flow'
    },

    {   "conditions": [
            lambda x: "ME" in x,
            lambda x: "(S)" in x,
            lambda x: "FLOW" in x,
        ],
        "action": 'ME1Flow'
    },


    {   "conditions": [
            lambda x: "ME" not in x,
            lambda x: "GE" not in x,
            lambda x: "(P)" not in x,
            lambda x: "(S)" not in x,
            lambda x: "FLOW" in x,
        ],
        "action": 'ME1Flow'
    },
    {   "conditions": [
            lambda x: "ME" in x,
            lambda x: "GE" not in x,
            lambda x: "(P)" not in x,
            lambda x: "(S)" not in x,
            lambda x: "FLOW" not in x,
            lambda x: "CONSUMPTION" in x,
        ],
        "action": 'ME1Flow'
    },
]

# GEFlow rules
GEFlow_rules = [
        {   "conditions": [
            lambda x: "NO" not in x,
            lambda x: "GE" in x,
            lambda x: "MGO" in x,
        ],
        "action": 'GE1Flow'
    },
    {   "conditions": [ 
            lambda x: "NO1" in x,
            lambda x: "GE" in x,
            lambda x: "MGO" in x,

        ],
        "action": 'GE1Flow'
    },

        {   "conditions": [ 
            lambda x: "NO2" in x,
            lambda x: "GE" in x,
            lambda x: "MGO" in x,

        ],
        "action": 'GE2Flow'
    },
        {   "conditions": [ 
            lambda x: "NO3" in x,
            lambda x: "GE" in x,
            lambda x: "MGO" in x,

        ],
        "action": 'GE3Flow'
    },

        {   "conditions": [ 
            lambda x: "NO" not in x,
            lambda x: "GE" in x,
            lambda x: "CONSUMPTION" in x,

        ],
        "action": 'GE1Flow'
    },
]