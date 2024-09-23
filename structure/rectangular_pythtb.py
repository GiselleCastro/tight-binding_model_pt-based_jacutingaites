from __future__ import print_function
from definitions.products import *
from pythtb import tb_model
import numpy as np
from MNX import *

def set_model(hop_second_N_N_soc,
              hop_first_Pt_Pt_soc,
              hop_second_Pt_Pt_soc,
              hop_second_Pt_N_soc
              ):

    [
        energy_on_site_N,
        hop_first_N_N,
        hop_second_N_N,
        energy_on_site_Pt,
        hop_first_Pt_Pt,
        hop_second_Pt_Pt,
        hop_first_Pt_N,
        hop_second_Pt_N
    ] = variables_wosoc

    hop_second_N_N_soc *= 1j
    hop_first_Pt_Pt_soc *= 1j
    hop_second_Pt_Pt_soc *= 1j
    hop_second_Pt_N_soc *= 1j

    lat = [
            [ 1.0*a, 0, 0], 
            [ 0, a*np.sqrt(3.0), 0], 
            [ 0, 0, c]
        ]
    
    orb = [
           [ 1.0     , 2/3 , X1[2]],
           [ 0.5     , 5/6 , X2[2]],
           [ 1.0     , 0.5 , 0.5  ],
           [ 0.25    , 0.75, 0.5  ],
           [ 0.75    , 0.75, 0.5  ],
           [ 0.5     , 1/6 , X1[2]],
           [ 0.0     , 1/3 , X2[2]],
           [ 0.5     , 0.0 , 0.5  ],
           [ 0.75 - 1, 0.25, 0.5  ],
           [ 0.25    , 0.25, 0.5  ]
           ]

    model = tb_model(2, 3, lat, orb, nspin=2)

    model.set_onsite([
        energy_on_site_N, 
        energy_on_site_N,
        energy_on_site_Pt, 
        energy_on_site_Pt, 
        energy_on_site_Pt, 
        energy_on_site_N, 
        energy_on_site_N,
        energy_on_site_Pt, 
        energy_on_site_Pt, 
        energy_on_site_Pt
        ])

    orb_lattice = []

    def lattice_vector(vector):
        return np.dot(np.array(vector), np.array([
                    [ 1.0, 0.0, 0], 
                    [ 0, np.sqrt(3.0), 0], 
                    [ 0, 0, c/a]
                ])
            )

    for unit_orb in np.array(orb):
        orb_lattice.append(lattice_vector(unit_orb))

    pos_orb = np.array(orb_lattice)

    # 1st N - N real
    for lvec in ([0, 0, 0], [1, 0, 0]):
        model.set_hop(-hop_first_N_N, 0, 1, lvec)
        model.set_hop(-hop_first_N_N, 5, 6, lvec)
    model.set_hop(-hop_first_N_N, 0, 6, [1, 0, 0])
    model.set_hop(-hop_first_N_N, 5, 1, [0,-1, 0])
    # 2nd N - N real
    for lavec in ([0, 0, 0], [0, 1, 0], [ 1, 0, 0], [ 1, 1, 0]):
        model.set_hop(-hop_second_N_N, 0, 5, lavec)
        model.set_hop(-hop_second_N_N, 1, 6, lavec)
    model.set_hop(-hop_second_N_N, 0, 0, [-1, 0, 0])
    model.set_hop(-hop_second_N_N, 1, 1, [-1, 0, 0])
    model.set_hop(-hop_second_N_N, 5, 5, [ 1, 0, 0])
    model.set_hop(-hop_second_N_N, 6, 6, [ 1, 0, 0])
    # 2nd N - N imaginary
    model.set_hop(hop_second_N_N_soc*array_matrix_Pauli(productVectorTwoNorm(pos_orb[0]-(pos_orb[6]+lattice_vector([ 1, 0, 0])),(pos_orb[6]+lattice_vector([ 1, 0, 0]))-(pos_orb[5]))), 0, 5, [ 0, 0, 0],mode='add')
    model.set_hop(hop_second_N_N_soc*array_matrix_Pauli(productVectorTwoNorm(pos_orb[1]-(pos_orb[0]+lattice_vector([-1, 0, 0])),(pos_orb[0]+lattice_vector([-1, 0, 0]))-(pos_orb[6]))), 1, 6, [ 0, 0, 0],mode='add')
    model.set_hop(hop_second_N_N_soc*array_matrix_Pauli(productVectorTwoNorm(pos_orb[0]-pos_orb[1], pos_orb[1]-(pos_orb[5]+lattice_vector([ 0, 1, 0])))), 0, 5, [ 0, 1, 0],mode='add')
    model.set_hop(hop_second_N_N_soc*array_matrix_Pauli(productVectorTwoNorm(pos_orb[1]-(pos_orb[5]+lattice_vector([ 0, 1, 0])),(pos_orb[5]+lattice_vector([ 0, 1, 0]))-(pos_orb[6]+lattice_vector([ 0, 1, 0])))), 1, 6, [ 0, 1, 0],mode='add')
    model.set_hop(hop_second_N_N_soc*array_matrix_Pauli(productVectorTwoNorm(pos_orb[0]- (pos_orb[6]+lattice_vector([ 1, 0, 0])), (pos_orb[6]+lattice_vector([ 1, 0, 0]))-(pos_orb[5]+lattice_vector([ 1, 0, 0])))), 0, 5, [ 1, 0, 0],mode='add')
    model.set_hop(hop_second_N_N_soc*array_matrix_Pauli(productVectorTwoNorm(pos_orb[1]- pos_orb[0], pos_orb[0]-(pos_orb[6]+lattice_vector([ 1, 0, 0])))), 1, 6, [ 1, 0, 0],mode='add')
    model.set_hop(hop_second_N_N_soc*array_matrix_Pauli(productVectorTwoNorm(pos_orb[0]-(pos_orb[1]+lattice_vector([ 1, 0, 0])),(pos_orb[1]+lattice_vector([ 1, 0, 0]))-(pos_orb[5] + lattice_vector([ 1, 1, 0])))), 0, 5, [ 1, 1, 0],mode='add')
    model.set_hop(hop_second_N_N_soc*array_matrix_Pauli(productVectorTwoNorm(pos_orb[1]-(pos_orb[5]+lattice_vector([ 0, 1, 0])),(pos_orb[5]+lattice_vector([ 0, 1, 0]))-(pos_orb[6] + lattice_vector([ 1, 1, 0])))), 1, 6, [ 1, 1, 0],mode='add')
    model.set_hop(hop_second_N_N_soc*array_matrix_Pauli(productVectorTwoNorm(pos_orb[0]-pos_orb[1], pos_orb[1]-(pos_orb[0]+lattice_vector([-1, 0, 0])))), 0, 0, [-1, 0, 0],mode='add')
    model.set_hop(hop_second_N_N_soc*array_matrix_Pauli(productVectorTwoNorm(pos_orb[1]-(pos_orb[0]+lattice_vector([-1, 0, 0])),(pos_orb[0]+lattice_vector([-1, 0, 0]))-(pos_orb[1]+lattice_vector([-1, 0, 0])))), 1, 1, [-1, 0, 0],mode='add')
    model.set_hop(hop_second_N_N_soc*array_matrix_Pauli(productVectorTwoNorm(pos_orb[5]- (pos_orb[6]+lattice_vector([ 1, 0, 0])), (pos_orb[6]+lattice_vector([ 1, 0, 0]))-(pos_orb[5]+lattice_vector([ 1, 0, 0])))), 5, 5, [ 1, 0, 0],mode='add')
    model.set_hop(hop_second_N_N_soc*array_matrix_Pauli(productVectorTwoNorm(pos_orb[6]- pos_orb[5], pos_orb[5]-(pos_orb[6]+lattice_vector([ 1, 0, 0])))), 6, 6, [ 1, 0, 0],mode='add')
    # 1st Pt - Pt real
    model.set_hop(-hop_first_Pt_Pt, 2 , 4, [ 0, 0, 0])
    model.set_hop(-hop_first_Pt_Pt, 3 , 4, [ 0, 0, 0])
    model.set_hop(-hop_first_Pt_Pt, 7 , 9, [ 0, 0, 0])
    model.set_hop(-hop_first_Pt_Pt, 8 , 9, [ 0, 0, 0])
    model.set_hop(-hop_first_Pt_Pt, 2 , 3, [ 1, 0, 0])
    model.set_hop(-hop_first_Pt_Pt, 2 , 8, [ 1, 0, 0])
    model.set_hop(-hop_first_Pt_Pt, 2 , 9, [ 1, 0, 0])
    model.set_hop(-hop_first_Pt_Pt, 3 , 4, [-1, 0, 0])
    model.set_hop(-hop_first_Pt_Pt, 7 , 8, [ 1, 0, 0])
    model.set_hop(-hop_first_Pt_Pt, 9 , 8, [ 1, 0, 0])
    model.set_hop(-hop_first_Pt_Pt, 7 , 3, [ 0,-1, 0])
    model.set_hop(-hop_first_Pt_Pt, 7 , 4, [ 0,-1, 0])
    # 1st Pt - Pt imaginary
    model.set_hop(hop_first_Pt_Pt_soc*array_matrix_Pauli(productVectorTwoNorm(pos_orb[2]- pos_orb[0],pos_orb[0]-pos_orb[4])), 2 , 4, [ 0, 0, 0], mode='add')
    model.set_hop(hop_first_Pt_Pt_soc*array_matrix_Pauli(productVectorTwoNorm(pos_orb[3]- pos_orb[1],pos_orb[1] - pos_orb[4])), 3 , 4, [ 0, 0, 0], mode='add')
    model.set_hop(hop_first_Pt_Pt_soc*array_matrix_Pauli(productVectorTwoNorm(pos_orb[7]- pos_orb[5],pos_orb[5]-pos_orb[9])), 7 , 9, [ 0, 0, 0], mode='add')
    model.set_hop(hop_first_Pt_Pt_soc*array_matrix_Pauli(productVectorTwoNorm(pos_orb[8]- pos_orb[6],pos_orb[6] - pos_orb[9])), 8 , 9, [ 0, 0, 0], mode='add')
    model.set_hop(hop_first_Pt_Pt_soc*array_matrix_Pauli(productVectorTwoNorm(pos_orb[2]- pos_orb[0],pos_orb[0]-(pos_orb[3]+lattice_vector([1, 0, 0])))), 2 , 3, [ 1, 0, 0], mode='add')
    model.set_hop(hop_first_Pt_Pt_soc*array_matrix_Pauli(productVectorTwoNorm(pos_orb[2]-(pos_orb[6]+lattice_vector([1, 0, 0])),(pos_orb[6]+lattice_vector([1, 0, 0]))-(pos_orb[8]+lattice_vector([1, 0, 0])))), 2 , 8, [ 1, 0, 0], mode='add')
    model.set_hop(hop_first_Pt_Pt_soc*array_matrix_Pauli(productVectorTwoNorm(pos_orb[2]-(pos_orb[6]+lattice_vector([1, 0, 0])),(pos_orb[6]+lattice_vector([1, 0, 0]))-(pos_orb[9]+lattice_vector([1, 0, 0])))), 2 , 9, [ 1, 0, 0], mode='add')
    model.set_hop(hop_first_Pt_Pt_soc*array_matrix_Pauli(productVectorTwoNorm(pos_orb[3]-(pos_orb[0]+lattice_vector([-1, 0, 0])),(pos_orb[0]+lattice_vector([-1, 0, 0]))-(pos_orb[4]+lattice_vector([-1, 0, 0])))), 3 , 4, [-1, 0, 0], mode='add')
    model.set_hop(hop_first_Pt_Pt_soc*array_matrix_Pauli(productVectorTwoNorm(pos_orb[7]- pos_orb[5],pos_orb[5]-(pos_orb[8]+lattice_vector([1, 0, 0])))), 7 , 8, [ 1, 0, 0], mode='add')
    model.set_hop(hop_first_Pt_Pt_soc*array_matrix_Pauli(productVectorTwoNorm(pos_orb[9]-pos_orb[5], pos_orb[5]-(pos_orb[8]+lattice_vector([ 1, 0, 0])))), 9 , 8, [ 1, 0, 0], mode='add')
    model.set_hop(hop_first_Pt_Pt_soc*array_matrix_Pauli(productVectorTwoNorm(pos_orb[7]-(pos_orb[1]+lattice_vector([0,-1, 0])),(pos_orb[1]+lattice_vector([0,-1, 0]))-(pos_orb[3]+lattice_vector([0,-1, 0])))), 7 , 3, [ 0,-1, 0], mode='add')
    model.set_hop(hop_first_Pt_Pt_soc*array_matrix_Pauli(productVectorTwoNorm(pos_orb[7]-(pos_orb[1]+lattice_vector([0,-1, 0])),(pos_orb[1]+lattice_vector([0,-1, 0]))-(pos_orb[4]+lattice_vector([0,-1, 0])))), 7 , 4, [ 0,-1, 0], mode='add')
    # 2nd Pt - Pt real
    model.set_hop(-hop_second_Pt_Pt, 2 , 3, [ 0, 0, 0])
    model.set_hop(-hop_second_Pt_Pt, 2 , 9, [ 0, 0, 0])
    model.set_hop(-hop_second_Pt_Pt, 3 , 9, [ 0, 0, 0])
    model.set_hop(-hop_second_Pt_Pt, 8 , 7, [ 0, 0, 0])
    model.set_hop(-hop_second_Pt_Pt, 2 , 8, [ 2, 0, 0])
    model.set_hop(-hop_second_Pt_Pt, 2 , 4, [ 1, 0, 0])
    model.set_hop(-hop_second_Pt_Pt, 3 , 9, [ 0, 1, 0])
    model.set_hop(-hop_second_Pt_Pt, 3 , 7, [-1, 1, 0])
    model.set_hop(-hop_second_Pt_Pt, 9,  7, [-1, 0, 0])
    model.set_hop(-hop_second_Pt_Pt, 4 , 8, [ 1, 0, 0])
    model.set_hop(-hop_second_Pt_Pt, 4 , 7, [ 1, 1, 0])
    model.set_hop(-hop_second_Pt_Pt, 4 , 8, [ 1, 1, 0])
    # 2nd Pt - Pt imaginary 
    model.set_hop(hop_second_Pt_Pt_soc * array_matrix_Pauli(productVectorFourNorm(pos_orb[2]-pos_orb[0],pos_orb[0]-pos_orb[4],pos_orb[4]-pos_orb[1],pos_orb[1]-pos_orb[3])), 2 , 3, [ 0, 0, 0], mode='add')
    model.set_hop(hop_second_Pt_Pt_soc * array_matrix_Pauli(productVectorFourNorm(pos_orb[2]-(pos_orb[6]+lattice_vector([ 1, 0, 0])),(pos_orb[6]+lattice_vector([ 1, 0, 0]))-(pos_orb[8]+lattice_vector([ 1, 0, 0])),(pos_orb[8]+lattice_vector([ 1, 0, 0]))-pos_orb[5],pos_orb[5]-pos_orb[9])), 2 , 9, [ 0, 0, 0], mode='add')
    model.set_hop(hop_second_Pt_Pt_soc * array_matrix_Pauli(productVectorFourNorm(pos_orb[3]-(pos_orb[0]+lattice_vector([-1, 0, 0])),(pos_orb[0]+lattice_vector([-1, 0, 0]))-(pos_orb[2]+lattice_vector([-1, 0, 0])),(pos_orb[2]+lattice_vector([-1, 0, 0]))-pos_orb[6],pos_orb[6]-pos_orb[9])), 3 , 9, [ 0, 0, 0], mode='add')
    model.set_hop(hop_second_Pt_Pt_soc * array_matrix_Pauli(productVectorFourNorm(pos_orb[8]-pos_orb[6], pos_orb[6]-pos_orb[9], pos_orb[9]-pos_orb[5],pos_orb[5]-pos_orb[7])), 8 , 7, [ 0, 0, 0], mode='add')
    model.set_hop(hop_second_Pt_Pt_soc * array_matrix_Pauli(productVectorFourNorm(pos_orb[2]-(pos_orb[6]+lattice_vector([ 1, 0, 0])),(pos_orb[6]+lattice_vector([ 1, 0, 0]))-(pos_orb[9]+lattice_vector([ 1, 0, 0])),(pos_orb[9]+lattice_vector([ 1, 0, 0])) -(pos_orb[5]+lattice_vector([ 1, 0, 0])),(pos_orb[5]+lattice_vector([ 1, 0, 0]))-(pos_orb[8]+lattice_vector([ 2, 0, 0])))), 2 , 8, [ 2, 0, 0], mode='add')
    model.set_hop(hop_second_Pt_Pt_soc * array_matrix_Pauli(productVectorFourNorm(pos_orb[2]- pos_orb[0], pos_orb[0]-(pos_orb[3]+lattice_vector([ 1, 0, 0])),(pos_orb[3]+lattice_vector([ 1, 0, 0]))-(pos_orb[1]+lattice_vector([ 1, 0, 0])),(pos_orb[1]+lattice_vector([ 1, 0, 0]))-(pos_orb[4]+lattice_vector([ 1, 0, 0])))), 2 , 4, [ 1, 0, 0], mode='add')
    model.set_hop(hop_second_Pt_Pt_soc * array_matrix_Pauli(productVectorFourNorm(pos_orb[3]-pos_orb[1],pos_orb[1]-(pos_orb[7]+lattice_vector([ 0, 1, 0])),(pos_orb[7]+lattice_vector([ 0, 1, 0])) -(pos_orb[5]+lattice_vector([ 0, 1, 0])),(pos_orb[5]+lattice_vector([ 0, 1, 0]))-(pos_orb[9]+lattice_vector([ 0, 1, 0])))), 3 , 9, [ 0, 1, 0], mode='add')
    model.set_hop(hop_second_Pt_Pt_soc * array_matrix_Pauli(productVectorFourNorm(pos_orb[3]-(pos_orb[0]+lattice_vector([-1, 0, 0])),(pos_orb[0]+lattice_vector([-1, 0, 0]))-(pos_orb[4]+lattice_vector([-1, 0, 0])),(pos_orb[4]+lattice_vector([-1, 0, 0]))-(pos_orb[1]+lattice_vector([-1, 0, 0])),(pos_orb[1]+lattice_vector([-1, 0, 0]))-(pos_orb[2]+lattice_vector([-1, 1, 0])))), 3 , 7, [-1, 1, 0], mode='add')
    model.set_hop(hop_second_Pt_Pt_soc * array_matrix_Pauli(productVectorFourNorm(pos_orb[9]-pos_orb[6],pos_orb[6]-pos_orb[8],pos_orb[8]-(pos_orb[5]+lattice_vector([-1, 0, 0])),(pos_orb[5]+lattice_vector([-1, 0, 0]))-(pos_orb[7]+lattice_vector([-1, 0, 0])))), 9,  7, [-1, 0, 0], mode='add')
    model.set_hop(hop_second_Pt_Pt_soc * array_matrix_Pauli(productVectorFourNorm(pos_orb[4]-pos_orb[0],pos_orb[0]-pos_orb[2],pos_orb[2]-(pos_orb[6]+lattice_vector([ 1, 0, 0])),(pos_orb[6]+lattice_vector([ 1, 0, 0]))-(pos_orb[8]+lattice_vector([ 1, 0, 0])))), 4 , 8, [ 1, 0, 0], mode='add')
    model.set_hop(hop_second_Pt_Pt_soc * array_matrix_Pauli(productVectorFourNorm(pos_orb[4]-pos_orb[0],pos_orb[0]-(pos_orb[3]+lattice_vector([ 1, 0, 0])),(pos_orb[3]+lattice_vector([ 1, 0, 0])) -(pos_orb[1]+lattice_vector([ 1, 0, 0])),(pos_orb[1]+lattice_vector([ 1, 0, 0]))-(pos_orb[7]+lattice_vector([ 1, 1, 0])))), 4 , 7, [ 1, 1, 0], mode='add')
    model.set_hop(hop_second_Pt_Pt_soc * array_matrix_Pauli(productVectorFourNorm(pos_orb[4]-pos_orb[1],pos_orb[1]-(pos_orb[7]+lattice_vector([ 0, 1, 0])),(pos_orb[7]+lattice_vector([ 0, 1, 0]))-(pos_orb[5]+lattice_vector([ 0, 1, 0])),(pos_orb[5]+lattice_vector([ 0, 1, 0]))-(pos_orb[8]+lattice_vector([ 1, 1, 0])))), 4 , 8, [ 1, 1, 0], mode='add')
    # 1st Pt - N real
    model.set_hop(-hop_first_Pt_N, 0, 2, [0, 0, 0])
    model.set_hop(-hop_first_Pt_N, 0, 3, [1, 0, 0])
    model.set_hop(-hop_first_Pt_N, 0, 4, [0, 0, 0])
    model.set_hop(-hop_first_Pt_N, 1, 7, [0, 1, 0])
    model.set_hop(-hop_first_Pt_N, 1, 3, [0, 0, 0])
    model.set_hop(-hop_first_Pt_N, 1, 4, [0, 0, 0])
    model.set_hop(-hop_first_Pt_N, 5, 7, [0, 0, 0])
    model.set_hop(-hop_first_Pt_N, 5, 8, [1, 0, 0])
    model.set_hop(-hop_first_Pt_N, 5, 9, [0, 0, 0])
    model.set_hop(-hop_first_Pt_N, 6, 2, [-1,0, 0])
    model.set_hop(-hop_first_Pt_N, 6, 8, [0, 0, 0])
    model.set_hop(-hop_first_Pt_N, 6, 9, [0, 0, 0])
    #  2nd Pt - N real
    model.set_hop(-hop_second_Pt_N, 5, 8, [ 0, 0, 0])
    model.set_hop(-hop_second_Pt_N, 5, 2, [ 0, 0, 0])
    model.set_hop(-hop_second_Pt_N, 6, 7, [ 0, 0, 0])
    model.set_hop(-hop_second_Pt_N, 0, 3, [ 0, 0, 0])
    model.set_hop(-hop_second_Pt_N, 1, 2, [ 0, 0, 0])
    model.set_hop(-hop_second_Pt_N, 6, 3, [ 0, 0, 0])
    model.set_hop(-hop_second_Pt_N, 0, 8, [ 1, 0, 0])
    model.set_hop(-hop_second_Pt_N, 0, 9, [ 1, 0, 0])
    model.set_hop(-hop_second_Pt_N, 5, 9, [ 1, 0, 0])
    model.set_hop(-hop_second_Pt_N, 0, 4, [ 1, 0, 0])
    model.set_hop(-hop_second_Pt_N, 4, 1, [ 1, 0, 0])
    model.set_hop(-hop_second_Pt_N, 6, 8, [ 1, 0, 0])
    model.set_hop(-hop_second_Pt_N, 0, 7, [ 1, 1, 0])
    model.set_hop(-hop_second_Pt_N, 1, 8, [ 1, 1, 0])
    model.set_hop(-hop_second_Pt_N, 1, 2, [-1, 0, 0])
    model.set_hop(-hop_second_Pt_N, 3, 1, [-1, 0, 0])
    model.set_hop(-hop_second_Pt_N, 5, 2, [-1, 0, 0])
    model.set_hop(-hop_second_Pt_N, 6, 7, [-1, 0, 0])
    model.set_hop(-hop_second_Pt_N, 6, 9, [-1, 0, 0])
    model.set_hop(-hop_second_Pt_N, 6, 4, [-1, 0, 0])
    model.set_hop(-hop_second_Pt_N, 0, 7, [ 0, 1, 0])
    model.set_hop(-hop_second_Pt_N, 1, 9, [ 0, 1, 0])
    model.set_hop(-hop_second_Pt_N, 5, 3, [ 0,-1, 0])
    model.set_hop(-hop_second_Pt_N, 5, 4, [ 0,-1, 0])
    # 2nd Pt - N imaginary
    model.set_hop(hop_second_Pt_N_soc*array_matrix_Pauli(productVectorTwoNorm(pos_orb[5]-pos_orb[6],pos_orb[6]-pos_orb[8])), 5, 8, [ 0, 0, 0] , mode='add')
    model.set_hop(hop_second_Pt_N_soc*array_matrix_Pauli(productVectorTwoNorm(pos_orb[5]-(pos_orb[6]+lattice_vector([ 1, 0, 0])),(pos_orb[6]+lattice_vector([ 1, 0, 0]))-pos_orb[2])), 5, 2, [ 0, 0, 0] , mode='add')
    model.set_hop(hop_second_Pt_N_soc*array_matrix_Pauli(productVectorTwoNorm(pos_orb[6]-pos_orb[5],pos_orb[5]-pos_orb[7])), 6, 7, [ 0, 0, 0] , mode='add')
    model.set_hop(hop_second_Pt_N_soc*array_matrix_Pauli(productVectorTwoNorm(pos_orb[0]-pos_orb[1],pos_orb[1]-pos_orb[3])), 0, 3, [ 0, 0, 0] , mode='add')
    model.set_hop(hop_second_Pt_N_soc*array_matrix_Pauli(productVectorTwoNorm(pos_orb[1]-pos_orb[0],pos_orb[0]-pos_orb[2])), 1, 2, [ 0, 0, 0] , mode='add')
    model.set_hop(hop_second_Pt_N_soc*array_matrix_Pauli(productVectorTwoNorm(pos_orb[6]-(pos_orb[0]+lattice_vector([-1, 0, 0])),(pos_orb[0]+lattice_vector([-1, 0, 0]))-pos_orb[3])), 6, 3, [ 0, 0, 0] , mode='add')
    model.set_hop(hop_second_Pt_N_soc*array_matrix_Pauli(productVectorTwoNorm(pos_orb[0]-(pos_orb[6]+lattice_vector([ 1, 0, 0])),(pos_orb[6]+lattice_vector([ 1, 0, 0])) -(pos_orb[8]+lattice_vector([ 1, 0, 0])))), 0, 8, [ 1, 0, 0] , mode='add')
    model.set_hop(hop_second_Pt_N_soc*array_matrix_Pauli(productVectorTwoNorm(pos_orb[0]-(pos_orb[6]+lattice_vector([ 1, 0, 0])),(pos_orb[6]+lattice_vector([ 1, 0, 0])) -(pos_orb[9]+lattice_vector([ 1, 0, 0])))), 0, 9, [ 1, 0, 0] , mode='add')
    model.set_hop(hop_second_Pt_N_soc*array_matrix_Pauli(productVectorTwoNorm(pos_orb[5]-(pos_orb[6]+lattice_vector([ 1, 0, 0])),(pos_orb[6]+lattice_vector([ 1, 0, 0])) -(pos_orb[9]+lattice_vector([ 1, 0, 0])))), 5, 9, [ 1, 0, 0] , mode='add')
    model.set_hop(hop_second_Pt_N_soc*array_matrix_Pauli(productVectorTwoNorm(pos_orb[0]-(pos_orb[1]+lattice_vector([ 1, 0, 0])),(pos_orb[1]+lattice_vector([ 1, 0, 0])) -(pos_orb[4]+lattice_vector([ 1, 0, 0])))), 0, 4, [ 1, 0, 0] , mode='add')
    model.set_hop(hop_second_Pt_N_soc*array_matrix_Pauli(productVectorTwoNorm(pos_orb[4]-pos_orb[0],pos_orb[0]-(pos_orb[1]+lattice_vector([ 1, 0, 0])))), 4, 1, [ 1, 0, 0] , mode='add')
    model.set_hop(hop_second_Pt_N_soc*array_matrix_Pauli(productVectorTwoNorm(pos_orb[6]-pos_orb[5],pos_orb[5]-(pos_orb[8]+lattice_vector([ 1, 0, 0])))), 6, 8, [ 1, 0, 0] , mode='add')
    model.set_hop(hop_second_Pt_N_soc*array_matrix_Pauli(productVectorTwoNorm(pos_orb[0]-(pos_orb[1]+lattice_vector([ 1, 0, 0])),(pos_orb[1]+lattice_vector([ 1, 0, 0])) -(pos_orb[7]+lattice_vector([ 1, 1, 0])))), 0, 7, [ 1, 1, 0] , mode='add')
    model.set_hop(hop_second_Pt_N_soc*array_matrix_Pauli(productVectorTwoNorm(pos_orb[1]-(pos_orb[5]+lattice_vector([ 0, 1, 0])),(pos_orb[5]+lattice_vector([ 0, 1, 0])) -(pos_orb[8]+lattice_vector([ 1, 1, 0])))), 1, 8, [ 1, 1, 0] , mode='add')
    model.set_hop(hop_second_Pt_N_soc*array_matrix_Pauli(productVectorTwoNorm(pos_orb[1]-(pos_orb[0]+lattice_vector([-1, 0, 0])),(pos_orb[0]+lattice_vector([-1, 0, 0])) -(pos_orb[2]+lattice_vector([-1, 0, 0])))), 1, 2, [-1, 0, 0] , mode='add')
    model.set_hop(hop_second_Pt_N_soc*array_matrix_Pauli(productVectorTwoNorm(pos_orb[4]-(pos_orb[0]+lattice_vector([-1, 0, 0])),(pos_orb[0]+lattice_vector([-1, 0, 0])) -(pos_orb[1]+lattice_vector([-1, 0, 0])))), 3, 1, [-1, 0, 0] , mode='add')
    model.set_hop(hop_second_Pt_N_soc*array_matrix_Pauli(productVectorTwoNorm(pos_orb[5]-pos_orb[6],pos_orb[6]-(pos_orb[2]+lattice_vector([-1, 0, 0])))), 5, 2, [-1, 0, 0] , mode='add')
    model.set_hop(hop_second_Pt_N_soc*array_matrix_Pauli(productVectorTwoNorm(pos_orb[6]-(pos_orb[5]+lattice_vector([-1, 0, 0])),(pos_orb[5]+lattice_vector([-1, 0, 0]))-(pos_orb[7]+lattice_vector([-1, 0, 0])))), 6, 7, [-1, 0, 0] , mode='add')
    model.set_hop(hop_second_Pt_N_soc*array_matrix_Pauli(productVectorTwoNorm(pos_orb[6]-(pos_orb[5]+lattice_vector([-1, 0, 0])),(pos_orb[5]+lattice_vector([-1, 0, 0]))-(pos_orb[9]+lattice_vector([-1, 0, 0])))), 6, 9, [-1, 0, 0] , mode='add')
    model.set_hop(hop_second_Pt_N_soc*array_matrix_Pauli(productVectorTwoNorm(pos_orb[6]-(pos_orb[0]+lattice_vector([-1, 0, 0])),(pos_orb[0]+lattice_vector([-1, 0, 0])) -(pos_orb[4]+lattice_vector([-1, 0, 0])))), 6, 4, [-1, 0, 0] , mode='add')
    model.set_hop(hop_second_Pt_N_soc*array_matrix_Pauli(productVectorTwoNorm(pos_orb[0]-pos_orb[1],pos_orb[1]-(pos_orb[7]+lattice_vector([ 0, 1, 0])))), 0, 7, [ 0, 1, 0] , mode='add')
    model.set_hop(hop_second_Pt_N_soc*array_matrix_Pauli(productVectorTwoNorm(pos_orb[1]-(pos_orb[5]+lattice_vector([ 0, 1, 0])),(pos_orb[5]+lattice_vector([ 0, 1, 0])) -(pos_orb[9]+lattice_vector([ 0, 1, 0])))), 1, 9, [ 0, 1, 0] , mode='add')
    model.set_hop(hop_second_Pt_N_soc*array_matrix_Pauli(productVectorTwoNorm(pos_orb[5]-(pos_orb[1]+lattice_vector([ 0,-1, 0])),(pos_orb[1]+lattice_vector([ 0,-1, 0]))- (pos_orb[3]+lattice_vector([ 0,-1, 0])))), 5, 3, [ 0,-1, 0] , mode='add')
    model.set_hop(hop_second_Pt_N_soc*array_matrix_Pauli(productVectorTwoNorm(pos_orb[5]-(pos_orb[1]+lattice_vector([ 0,-1, 0])),(pos_orb[1]+lattice_vector([ 0,-1, 0])) -(pos_orb[4]+lattice_vector([ 0,-1, 0])))), 5, 4, [ 0,-1, 0] , mode='add')
    return model

