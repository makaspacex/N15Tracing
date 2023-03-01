def dcdt_func(t, c, p):
    k_kinetics = p.k_kinetics

    r1 = p.k0 * c.xN2 if k_kinetics[0] == 1 else p.k0
    r2 = p.k1 * c.xNH3 if k_kinetics[1] == 1 else p.k1
    r3 = p.k2 * c.xNO2 if k_kinetics[2] == 1 else p.k2
    r4 = p.k3 * c.xNO3 if k_kinetics[3] == 1 else p.k3
    r5 = p.k4 * c.xNO2 if k_kinetics[4] == 1 else p.k4
    r6 = p.k5 * c.xNO2 * c.xNO3 if k_kinetics[5] == 1 else p.k5
    r7 = p.k6 * c.xNO3 if k_kinetics[6] == 1 else p.k6
    r8 = p.k7 * c.xNO3 if k_kinetics[7] == 1 else p.k7
    r9 = p.k8 * c.xNH3 if k_kinetics[8] == 1 else p.k8
    r10 = p.k9 * c.xNOrg if k_kinetics[9] == 1 else p.k9
    r11 = p.k10 * c.xNOrg if k_kinetics[10] == 1 else p.k10

    dc_xNH3 = 2 * r1 + r7 + r10 - r2 - r6 - r9
    dc_xNO3 = r3 - r7 - r4 - r8 + r11
    dc_xNO2 = r2 + r4 - r3 - r6 - 2 * r5
    dc_xNOrg = r8 + r9 - r10 - r11
    dc_xN2 = r5 + r6 - r1

    dc_ANH3 = (2 * r1 * (c.AN2 - c.ANH3) + (c.ANO3 - c.ANH3) * r7 + (c.ANOrg - c.ANH3) * r10) / c.xNH3
    dc_ANO3 = ((c.ANO2 - c.ANO3) * r2 + (c.ANOrg - c.ANO3) * r11) / c.xNO3
    dc_ANO2 = ((c.ANH3 - c.ANO2) * r2 + (c.ANO3 - c.ANO2) * r4) / c.xNO2
    dc_ANOrg = ((c.ANO3 - c.ANOrg) * r8 + (c.ANH3 - c.ANOrg) * r9) / c.xNOrg
    dc_AN2 = ((c.ANO2 - c.AN2) * r5 + (c.ANO2 * c.ANH3 - c.AN2) * r6) / c.xN2

    # dcdts = [dc_xNH3, dc_xNO3, dc_xNO2, dc_xNOrg, dc_xN2, dc_ANH3, dc_ANO3, dc_ANO2, dc_ANOrg, dc_AN2]

    dcdts =  {
        'xNH3': dc_xNH3,
        'xNO3': dc_xNO3,
        'xNO2': dc_xNO2,
        'xNOrg': dc_xNOrg,
        'xN2': dc_xN2,
        'ANH3': dc_ANH3,
        'ANO3': dc_ANO3,
        'ANO2': dc_ANO2,
        'ANOrg': dc_ANOrg,
        'AN2': dc_AN2,
    }

    return dcdts
