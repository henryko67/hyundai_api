# substitution mapping for descriptions
# Abbreviations and their replacements
desc_replacement_dict = {
    r'\bLIST\b': 'LIST',
    # exhaust gas
    r'\bE\. GAS\b': 'EXHAUST GAS',
    r'\bEXH\.\b': 'EXHAUST',
    r'\bEXH\b': 'EXHAUST',
    r'\bEXHAUST\.\b': 'EXHAUST',
    r'\bEXHAUST\b': 'EXHAUST',
    r'\bBLR\.EXH\.\b': 'BOILER EXHAUST',
    # temperature
    r'\bTEMP\.\b': 'TEMPERATURE',
    r'\bTEMP\b': 'TEMPERATURE',
    r'\bTEMPERATURE\.\b': 'TEMPERATURE',
    r'\bTEMPERATURE\b': 'TEMPERATURE',
    # cylinder
    r'\bCYL(\d+)\b': r'CYLINDER\1',
    r'\bCYL\.(\d+)\b': r'CYLINDER\1',
    r'\bCYL(?=\d|\W|$)\b': 'CYLINDER',
    r'\bCYL\.\b': 'CYLINDER',
    r'\bCYL\b': 'CYLINDER',
    # cooling
    r'\bCOOL\.\b': 'COOLING',
    r'\bCOOLING\b': 'COOLING',
    r'\bCOOLER\b': 'COOLER',
    r'\bCW\b': 'COOLING WATER',
    r'\bC\.W\b': 'COOLING WATER',
    r'\bJ\.C\.F\.W\b': 'JACKET COOLING FEED WATER',
    r'\bJ\.C F\.W\b': 'JACKET COOLING FEED WATER',
    r'\bJACKET C\.F\.W\b': 'JACKET COOLING FEED WATER',
    r'\bCOOL\. F\.W\b': 'COOLING FEED WATER',
    r'\bC\.F\.W\b': 'COOLING FEED WATER',
    # sea water
    r'\bC\.S\.W\b': 'COOLING SEA WATER',
    r'\bCSW\b': 'COOLING SEA WATER',
    r'\bC.S.W\b': 'COOLING SEA WATER',
    # water
    r'\bFEED W\.\b': 'FEED WATER',
    r'\bFEED W\b': 'FEED WATER',
    r'\bF\.W\b': 'FEED WATER',
    r'\bF\.W\.\b': 'FEED WATER',
    r'\bFW\b': 'FEED WATER',
    # r'\bWATER\b': 'WATER',
    r'\bSCAV\.\b': 'SCAVENGE',
    r'\bSCAV\b': 'SCAVENGE',
    r'\bINL\.\b': 'INLET',
    r'\bINLET\b': 'INLET',
    r'\bOUT\.\b': 'OUTLET',
    r'\bOUTL\.\b': 'OUTLET',
    r'\bOUTLET\b': 'OUTLET',
    # tank
    r'\bSTOR\.TK\b': 'STORAGE TANK',
    r'\bSTOR\. TK\b': 'STORAGE TANK',
    r'\bSERV\. TK\b': 'SERVICE TANK',
    r'\bSETT\. TK\b': 'SETTLING TANK',
    r'\bBK\b': 'BUNKER',
    r'\bTK\b': 'TANK',
    # PRESSURE
    r'\bPRESS\b': 'PRESSURE',
    r'\bPRESS\.\b': 'PRESSURE',
    r'\bPRESSURE\b': 'PRESSURE',
    r'PRS\b': 'PRESSURE',  # this is a special replacement - it is safe to replace PRS w/o checks
    # ENGINE
    r'\bENG\.\b': 'ENGINE',
    r'\bENG\b': 'ENGINE',
    r'\bENGINE\b': 'ENGINE',
    r'\bENGINE SPEED\b': 'ENGINE SPEED',
    r'\bENGINE RUNNING\b': 'ENGINE RUNNING',
    r'\bENGINE RPM PICKUP\b': 'ENGINE RPM PICKUP',
    r'\bENGINE ROOM\b': 'ENGINE ROOM',
    r'\bE/R\b': 'ENGINE ROOM',
    # MAIN ENGINE
    r'\bM/E NO.(\d+)\b': r'NO\1 MAIN_ENGINE',
    r'\bM/E NO(\d+)\b': r'NO\1 MAIN_ENGINE',
    r'\bM/E  NO.(\d+)\b': r'NO\1 MAIN_ENGINE',
    r'\bME NO.(\d+)\b': r'NO\1 MAIN_ENGINE',
    r'\bM/E\b': 'MAIN_ENGINE',
    r'\bM/E(.)\b': r'MAIN_ENGINE \1', # M/E(S/P)
    r'\bME(.)\b': r'MAIN_ENGINE \1', # ME(S/P)
    r'\bM_E\b': 'MAIN_ENGINE',
    r'\bME(?=\d|\W|$)\b': 'MAIN_ENGINE',
    r'\bMAIN ENGINE\b': 'MAIN_ENGINE',
    # ENGINE variants
    r'\bM_E_RPM\b': 'MAIN ENGINE RPM',
    r'\bM/E_M\.G\.O\.\b': 'MAIN ENGINE MARINE GAS OIL',
    r'\bM/E_H\.F\.O\.\b': 'MAIN ENGINE HEAVY FUEL OIL',
    # GENERATOR ENGINE
    r'\bGEN(\d+)\b': r'NO\1 GENERATOR_ENGINE',
    r'\bGE(\d+)\b': r'NO\1 GENERATOR_ENGINE',
    # ensure that we substitute only for terms where following GE is num or special
    r'\bGE(?=\d|\W|$)\b': 'GENERATOR_ENGINE',
    r'\bG/E(\d+)\b': r'NO\1 GENERATOR_ENGINE',
    r'\bG/E\b': r'GENERATOR_ENGINE',
    r'\bG_E(\d+)\b': r'NO\1 GENERATOR_ENGINE',
    r'\bG_E\b': 'GENERATOR_ENGINE',
    r'\bGENERATOR ENGINE\b': 'GENERATOR_ENGINE',
    r'\bG/E_M\.G\.O\b': 'GENERATOR_ENGINE MARINE GAS OIL',
    # DG
    r'\bDG(\d+)\b': r'NO\1 GENERATOR_ENGINE',
    r'\bDG\b': 'GENERATOR_ENGINE',
    r'\bD/G\b': 'GENERATOR_ENGINE',
    r'\bDG(\d+)\((.)\)\b': r'NO\1\2 GENERATOR_ENGINE', # handle DG2(A)
    r'\bDG(\d+[A-Za-z])\b': r'NO\1 GENERATOR_ENGINE', # handle DG2A
    # DG variants
    r'\bDG_CURRENT\b': 'GENERATOR_ENGINE CURRENT',
    r'\bDG_LOAD\b': 'GENERATOR_ENGINE LOAD',
    r'\bDG_FREQUENCY\b': 'GENERATOR_ENGINE FREQUENCY',
    r'\bDG_VOLTAGE\b': 'GENERATOR_ENGINE VOLTAGE',
    r'\bDG_CLOSED\b': 'GENERATOR_ENGINE CLOSED',
    r'\bD/G_CURRENT\b': 'GENERATOR_ENGINE CURRENT',
    r'\bD/G_LOAD\b': 'GENERATOR_ENGINE LOAD',
    r'\bD/G_FREQUENCY\b': 'GENERATOR_ENGINE FREQUENCY',
    r'\bD/G_VOLTAGE\b': 'GENERATOR_ENGINE VOLTAGE',
    r'\bD/G_CLOSED\b': 'GENERATOR_ENGINE CLOSED',
    # MGE
    r'\b(\d+)MGE\b': r'NO\1 MAIN_GENERATOR_ENGINE',
    # generator engine and mgo
    r'\bG/E_M\.G\.O\.\b': r'GENERATOR_ENGINE MARINE GAS OIL',
    r'\bG/E_H\.F\.O\.\b': r'GENERATOR_ENGINE HEAVY FUEL OIL',
    # ultra low sulfur fuel oil
    r'\bU\.L\.S\.F\.O\b': 'ULTRA LOW SULFUR FUEL OIL',
    r'\bULSFO\b': 'ULTRA LOW SULFUR FUEL OIL',
    # marine gas oil
    r'\bM\.G\.O\b': 'MARINE GAS OIL',
    r'\bMGO\b': 'MARINE GAS OIL',
    r'\bMDO\b': 'MARINE DIESEL OIL',
    # light fuel oil
    r'\bL\.F\.O\b': 'LIGHT FUEL OIL',
    r'\bLFO\b': 'LIGHT FUEL OIL',
    # heavy fuel oil
    r'\bHFO\b': 'HEAVY FUEL OIL',
    r'\bH\.F\.O\b': 'HEAVY FUEL OIL',
    # piston cooling oil
    r'\bPCO\b': 'PISTON COOLING OIL',
    r'\bP\.C\.O\.\b': 'PISTON COOLING OIL',
    r'\bP\.C\.O\b': 'PISTON COOLING OIL',
    r'PISTION C.O': 'PISTON COOLING OIL',
    # diesel oil
    r'\bD.O\b': 'DIESEL OIL',
    # for remaining fuel oil that couldn't be substituted
    r'\bF\.O\b': 'FUEL OIL',
    r'\bFO\b': 'FUEL OIL',
    # lubricant
    r'\bLUB\.\b': 'LUBRICANT',
    r'\bLUBE\b': 'LUBRICANT',
    r'\bLUBR\.\b': 'LUBRICANT',
    r'\bLUBRICATING\.\b': 'LUBRICANT',
    r'\bLUBRICATION\.\b': 'LUBRICANT',
    # lubricating oil
    r'\bL\.O\b': 'LUBRICATING OIL',
    r'\bLO\b': 'LUBRICATING OIL',
    # lubricating oil pressure
    r'\bLO_PRESS\b': 'LUBRICATING OIL PRESSURE',
    r'\bLO_PRESSURE\b': 'LUBRICATING OIL PRESSURE',
    # temperature
    r'\bL\.T\b': 'LOW TEMPERATURE',
    r'\bLT\b': 'LOW TEMPERATURE',
    r'\bH\.T\b': 'HIGH TEMPERATURE',
    r'\bHT\b': 'HIGH TEMPERATURE',
    # BOILER
    # auxiliary boiler
    # replace these first before replacing AUXILIARY only
    r'\bAUX\.BOILER\b': 'AUXILIARY BOILER',
    r'\bAUX\. BOILER\b': 'AUXILIARY BOILER',
    r'\bAUX BLR\b': 'AUXILIARY BOILER',
    r'\bAUX\.\b': 'AUXILIARY',
    r'\bAUX\b': 'AUXILIARY',
    # composite boiler
    r'\bCOMP\. BOILER\b': 'COMPOSITE BOILER',
    r'\bCOMP\.BOILER\b': 'COMPOSITE BOILER',
    r'\bCOMP BOILER\b': 'COMPOSITE BOILER',
    r'\bCOMP\b': 'COMPOSITE',
    r'\bCMPS\b': 'COMPOSITE',
    # any other boiler
    r'\bBLR\.\b': 'BOILER',
    r'\bBLR\b': 'BOILER',
    r'\bBOILER W.CIRC.P/P\b': 'BOILER WATER CIRC P/P',
    # windind
    r'\bWIND\.\b': 'WINDING',
    r'\bWINDING\b': 'WINDING',
    # VOLTAGE/FREQ/CURRENT
    r'\bVLOT\.': 'VOLTAGE', # correct spelling
    r'\bVOLT\.': 'VOLTAGE',
    r'\bVOLTAGE\b': 'VOLTAGE',
    r'\bFREQ\.': 'FREQUENCY',
    r'\bFREQUENCY\b': 'FREQUENCY',
    r'\bCURR\.': 'CURRENT',
    r'\bCURRENT\b': 'CURRENT',
    # TURBOCHARGER
    r'\bTCA\b': 'TURBOCHARGER',
    r'\bTCB\b': 'TURBOCHARGER',
    r'\bT/C\b': 'TURBOCHARGER',
    r'\bT_C\b': 'TURBOCHARGER',
    r'\bT/C_RPM\b': 'TURBOCHARGER RPM',
    r'\bTC(\d+)\b': r'TURBOCHARGER\1',
    r'\bT/C(\d+)\b': r'TURBOCHARGER\1',
    r'\bTC(?=\d|\W|$)\b': 'TURBOCHARGER',
    r'\bTURBOCHAGER\b': 'TURBOCHARGER',
    r'\bTURBOCHARGER\b': 'TURBOCHARGER',
    r'\bTURBOCHG\b': 'TURBOCHARGER',
    # misc spelling errors
    r'\bOPERATOIN\b': 'OPERATION',
    # wrongly attached terms
    r'\bBOILERMGO\b': 'BOILER MGO',
    # additional standardizing replacement
    # replace # followed by a number with NO
    r'#(?=\d)\b': 'NO',
    r'\bNO\.(?=\d)\b': 'NO',
    r'\bNO\.\.(?=\d)\b': 'NO',
    # others:
    # generator
    r'\bGEN\.\b': 'GENERATOR',
    # others
    r'\bGEN\.WIND\.TEMP\b': 'GENERATOR WINDING TEMPERATURE',
    r'\bFLTR\b': 'FILTER',
    r'\bCLR\b': 'CLEAR',
}

# substitution mapping for units
# Abbreviations and their replacements
unit_replacement_dict = {
    r'\b%\b': 'PERCENT',
    r'\b-\b': '',
    r'\b-  \b': '',
    # ensure no character after A
    r'\bA(?!\w|/)': 'CURRENT',
    r'\bAmp(?!\w|/)': 'CURRENT',
    r'\bHz\b': 'HERTZ',
    r'\bKG/CM2\b': 'PRESSURE',
    r'\bKG/H\b': 'KILOGRAM PER HOUR',
    r'\bKNm\b': 'RPM',
    r'\bKW\b': 'POWER',
    r'\bKg(?!\w|/)': 'MASS',
    r'\bKw\b': 'POWER',
    r'\bL(?!\w|/)': 'VOLUME',
    r'\bMT/h\b': 'METRIC TONNES PER HOUR',
    r'\bMpa\b': 'PRESSURE',
    r'\bPF\b': 'POWER FACTOR',
    r'\bRPM\b': 'RPM',
    r'\bV(?!\w|/)': 'VOLTAGE',
    r'\bbar(?!\w|/)': 'PRESSURE',
    r'\bbarA\b': 'SCAVENGE PRESSURE',
    r'\bcST\b': 'VISCOSITY',
    r'\bcSt\b': 'VISCOSITY',
    r'\bcst\b': 'VISCOSITY',
    r'\bdeg(?!\w|/|\.)': 'DEGREE',
    r'\bdeg.C\b': 'TEMPERATURE',
    r'\bdegC\b': 'TEMPERATURE',
    r'\bdegree\b': 'DEGREE',
    r'\bdegreeC\b': 'TEMPERATURE',
    r'\bhPa\b': 'PRESSURE',
    r'\bhours\b': 'HOURS',
    r'\bkN\b': 'THRUST',
    r'\bkNm\b': 'TORQUE',
    r'\bkW\b': 'POWER',
    # ensure that kg is not followed by anything
    r'\bkg(?!\w|/)': 'FLOW', # somehow in the data its flow
    r'\bkg/P\b': 'MASS FLOW',
    r'\bkg/cm2\b': 'PRESSURE',
    r'\bkg/cm²\b': 'PRESSURE',
    r'\bkg/h\b': 'MASS FLOW',
    r'\bkg/hr\b': 'MASS FLOW',
    r'\bkg/pulse\b': '',
    r'\bkgf/cm2\b': 'PRESSURE',
    r'\bkgf/cm²\b': 'PRESSURE',
    r'\bkgf/㎠\b': 'PRESSURE',
    r'\bknots\b': 'SPEED',
    r'\bkw\b': 'POWER',
    r'\bl/Hr\b': 'VOLUME FLOW',
    r'\bl/h\b': 'VOLUME FLOW',
    r'\bl_Hr\b': 'VOLUME FLOW',
    r'\bl_hr\b': 'VOLUME FLOW',
    r'\bM\b': 'DRAFT', # for wind draft
    r'm': 'm', # wind draft and trim - not useful
    r'\bm/s\b': 'SPEED',
    r'\bm3\b': 'VOLUME',
    r'\bmH2O\b': 'DRAFT',
    r'\bmWC\b': 'DRAFT',
    r'\bmbar\b': 'PRESSURE',
    r'\bmg\b': 'ACCELERATION',
    r'\bmin-¹\b': '', # data too varied
    r'\bmm\b': '', # data too varied
    r'\bmmH2O\b': 'WATER DRUM LEVEL',
    r'\brev\b': 'RPM',
    r'\brpm\b': 'RPM',
    r'\bx1000min-¹\b': '',
    r'\b°C\b': 'TEMPERATURE',
    r'\bºC\b': 'TEMPERATURE',
    r'\b℃\b': 'TEMPERATURE'
}