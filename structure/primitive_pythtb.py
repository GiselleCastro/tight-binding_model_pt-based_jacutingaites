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
            [-0.5*a, a*np.sqrt(3.0)/2.0, 0], 
            [ 0, 0, c]
        ]
    
    orb = [
            X1, 
            X2,
            [0.5, 0.0, 0.5], 
            [0.0, 0.5, 0.5], 
            [0.5, 0.5, 0.5]
        ]
    
    model = tb_model(2, 3, lat, orb, nspin=2)

    model.set_onsite([
        energy_on_site_N, 
        energy_on_site_N,
        energy_on_site_Pt, 
        energy_on_site_Pt, 
        energy_on_site_Pt
        ])

    orb_lattice = []

    def lattice_vector(vector):
        return np.dot(np.array(vector), 
                      np.array([
                          [1.0, 0.0, 0], 
                          [-0.5, np.sqrt(3.0)/2.0, 0], 
                          [0, 0, c/a]
                        ])
                    )

    for unit_orb in np.array(orb):
        orb_lattice.append(np.dot(unit_orb, np.array([
                    [1.0, 0.0, 0], 
                    [-0.5, np.sqrt(3.0)/2.0, 0], 
                    [0, 0, c/a]
                    ])
                ))

    pos_orb = np.array(orb_lattice)

    # 1st N - N real
    for lvec in ([ 0, 0, 0], [ 1, 0, 0], [ 0, -1, 0]):
        model.set_hop(-hop_first_N_N, 0, 1, lvec)

    # 2nd N - N real
    for lavec in ([ 0, 1, 0], [ 1, 0, 0], [-1, -1, 0]):
        model.set_hop(-hop_second_N_N, 0, 0, lavec)
        model.set_hop(-hop_second_N_N, 1, 1, lavec)

    # 2nd N - N imaginary
    model.set_hop(hop_second_N_N_soc
                  * array_matrix_Pauli(productVectorTwoNorm(
                      pos_orb[0] - pos_orb[1],
                      pos_orb[1] - (pos_orb[0] + lattice_vector([ 0, 1, 0]))
                  )), 0, 0, [ 0, 1, 0], mode='add')
    model.set_hop(hop_second_N_N_soc
                  * array_matrix_Pauli(productVectorTwoNorm(
                      pos_orb[0] - (pos_orb[1] + lattice_vector([ 0,-1, 0])),
                      (pos_orb[1] + lattice_vector([ 0,-1, 0])) -
                      (pos_orb[0] + lattice_vector([-1,-1, 0]))
                  )), 0, 0, [-1, -1, 0], mode='add')
    model.set_hop(hop_second_N_N_soc
                  * array_matrix_Pauli(productVectorTwoNorm(
                      pos_orb[0] - (pos_orb[1] + lattice_vector([ 1, 0, 0])),
                      (pos_orb[1] + lattice_vector([ 1, 0, 0])) -
                      (pos_orb[0] + lattice_vector([ 1, 0, 0]))
                  )), 0, 0, [ 1, 0, 0], mode='add')
    model.set_hop(hop_second_N_N_soc
                  * array_matrix_Pauli(productVectorTwoNorm(
                      pos_orb[1] - (pos_orb[0] + lattice_vector([ 0, 1, 0])),
                      (pos_orb[0] + lattice_vector([ 0, 1, 0])) -
                      (pos_orb[1] + lattice_vector([ 0, 1, 0]))
                  )), 1, 1, [ 0, 1, 0], mode='add')
    model.set_hop(hop_second_N_N_soc
                  * array_matrix_Pauli(productVectorTwoNorm(
                      pos_orb[1] - (pos_orb[0] + lattice_vector([-1, 0, 0])),
                      ((pos_orb[0] + lattice_vector([-1, 0, 0]))) -
                      (pos_orb[1] + lattice_vector([-1, -1, 0]))
                  )), 1, 1, [-1, -1, 0], mode='add')
    model.set_hop(hop_second_N_N_soc
                  * array_matrix_Pauli(productVectorTwoNorm(
                      pos_orb[1] - pos_orb[0],
                      pos_orb[0] - (pos_orb[1] + lattice_vector([ 1, 0, 0]))
                  )), 1, 1, [ 1, 0, 0], mode='add')

    # 1st Pt - Pt real
    model.set_hop(-hop_first_Pt_Pt, 2, 4, [ 0, 0, 0])
    model.set_hop(-hop_first_Pt_Pt, 2, 3, [ 1, 0, 0])
    model.set_hop(-hop_first_Pt_Pt, 2, 3, [ 0,-1, 0])
    model.set_hop(-hop_first_Pt_Pt, 2, 4, [ 0,-1, 0])
    model.set_hop(-hop_first_Pt_Pt, 3, 4, [ 0, 0, 0])
    model.set_hop(-hop_first_Pt_Pt, 3, 4, [-1, 0, 0])

    # 1st Pt - Pt imaginary
    model.set_hop(hop_first_Pt_Pt_soc
                  * array_matrix_Pauli(productVectorTwoNorm(
                      pos_orb[2] - pos_orb[0],
                      pos_orb[0] - pos_orb[4]
                  )), 2, 4, [ 0, 0, 0], mode='add')
    model.set_hop(hop_first_Pt_Pt_soc
                  * array_matrix_Pauli(productVectorTwoNorm(
                      pos_orb[2] - pos_orb[0],
                      pos_orb[0] - (pos_orb[3] + lattice_vector([ 1, 0, 0]))
                  )), 2, 3, [ 1, 0, 0], mode='add')
    model.set_hop(hop_first_Pt_Pt_soc
                  * array_matrix_Pauli(productVectorTwoNorm(
                      pos_orb[2]-(pos_orb[1] + lattice_vector([ 0,-1, 0])),
                      (pos_orb[1] + lattice_vector([ 0,-1, 0])) -
                      (pos_orb[3] + lattice_vector([ 0,-1, 0]))
                  )), 2, 3, [ 0,-1, 0], mode='add')
    model.set_hop(hop_first_Pt_Pt_soc
                  * array_matrix_Pauli(productVectorTwoNorm(
                      pos_orb[2]-(pos_orb[1] + lattice_vector([ 0,-1, 0])),
                      (pos_orb[1] + lattice_vector([ 0,-1, 0])) -
                      (pos_orb[4] + lattice_vector([ 0,-1, 0]))
                  )), 2, 4, [ 0,-1, 0], mode='add')
    model.set_hop(hop_first_Pt_Pt_soc
                  * array_matrix_Pauli(productVectorTwoNorm(
                      pos_orb[3] - pos_orb[1],
                      pos_orb[1] - pos_orb[4]
                  )), 3, 4, [ 0, 0, 0], mode='add')
    model.set_hop(hop_first_Pt_Pt_soc
                  * array_matrix_Pauli(productVectorTwoNorm(
                      pos_orb[3]-(pos_orb[0] + lattice_vector([-1, 0, 0])),
                      (pos_orb[0] + lattice_vector([-1, 0, 0])) -
                      (pos_orb[4] + lattice_vector([-1, 0, 0]))
                  )), 3, 4, [-1, 0, 0], mode='add')

    # 2nd Pt - Pt real
    model.set_hop(-hop_second_Pt_Pt, 2, 3, [ 0, 0, 0])
    model.set_hop(-hop_second_Pt_Pt, 2, 4, [ 1, 0, 0])
    model.set_hop(-hop_second_Pt_Pt, 2, 3, [ 1,-1, 0])
    model.set_hop(-hop_second_Pt_Pt, 2, 4, [-1,-1, 0])
    model.set_hop(-hop_second_Pt_Pt, 3, 4, [ 0, 1, 0])
    model.set_hop(-hop_second_Pt_Pt, 3, 4, [-1,-1, 0])

    # 2nd Pt - Pt imaginary 
    model.set_hop(hop_second_Pt_Pt_soc
                  * array_matrix_Pauli(productVectorFourNorm(
                      pos_orb[2]-pos_orb[0],
                      pos_orb[0]-pos_orb[4],
                      pos_orb[4]-pos_orb[1],
                      pos_orb[1]-pos_orb[3]
                  )), 2, 3, [ 0, 0, 0], mode='add')
    model.set_hop(hop_second_Pt_Pt_soc
                  * array_matrix_Pauli(productVectorFourNorm(
                      pos_orb[2]-pos_orb[0],
                      pos_orb[0]-(pos_orb[3]+lattice_vector([ 1, 0, 0])),
                      (pos_orb[3]+lattice_vector([ 1, 0, 0])) -
                      (pos_orb[1]+lattice_vector([ 1, 0, 0])),
                      (pos_orb[1]+lattice_vector([ 1, 0, 0])) -
                      (pos_orb[4]+lattice_vector([ 1, 0, 0]))
                  )), 2, 4, [ 1, 0, 0], mode='add')
    model.set_hop(hop_second_Pt_Pt_soc
                  * array_matrix_Pauli(productVectorFourNorm(
                      pos_orb[2]-(pos_orb[1]+lattice_vector([ 0,-1, 0])),
                      (pos_orb[1]+lattice_vector([ 0, -1, 0])) -
                      (pos_orb[4]+lattice_vector([ 0, -1, 0])),
                      (pos_orb[4]+lattice_vector([ 0, -1, 0])) -
                      (pos_orb[0]+lattice_vector([ 0, -1, 0])),
                      (pos_orb[0]+lattice_vector([ 0, -1, 0])) -
                      (pos_orb[3]+lattice_vector([ 1, -1, 0]))
                  )), 2, 3, [ 1,-1, 0], mode='add')
    model.set_hop(hop_second_Pt_Pt_soc
                  * array_matrix_Pauli(productVectorFourNorm(
                      pos_orb[2]-(pos_orb[1]+lattice_vector([ 0,-1, 0])),
                      (pos_orb[1]+lattice_vector([ 0,-1, 0])) -
                      (pos_orb[3]+lattice_vector([ 0,-1, 0])),
                      (pos_orb[3]+lattice_vector([ 0,-1, 0])) -
                      (pos_orb[0]+lattice_vector([-1,-1, 0])),
                      (pos_orb[0]+lattice_vector([-1,-1, 0])) -
                      (pos_orb[4]+lattice_vector([-1,-1, 0]))
                  )), 2, 4, [-1,-1, 0], mode='add')
    model.set_hop(hop_second_Pt_Pt_soc
                  * array_matrix_Pauli(productVectorFourNorm(
                      pos_orb[3]-pos_orb[1],
                      pos_orb[1]-(pos_orb[2]+lattice_vector([ 0, 1, 0])),
                      (pos_orb[2]+lattice_vector([ 0, 1, 0])) -
                      (pos_orb[0]+lattice_vector([ 0, 1, 0])),
                      (pos_orb[0]+lattice_vector([ 0, 1, 0])) -
                      (pos_orb[4]+lattice_vector([ 0, 1, 0]))
                  )), 3, 4, [ 0, 1, 0], mode='add')
    model.set_hop(hop_second_Pt_Pt_soc
                  * array_matrix_Pauli(productVectorFourNorm(
                      pos_orb[3]-(pos_orb[0]+lattice_vector([-1, 0, 0])),
                      (pos_orb[0]+lattice_vector([-1, 0, 0])) -
                      (pos_orb[2]+lattice_vector([-1, 0, 0])),
                      (pos_orb[2]+lattice_vector([-1, 0, 0])) -
                      (pos_orb[1]+lattice_vector([-1,-1, 0])),
                      (pos_orb[1]+lattice_vector([-1,-1, 0])) -
                      (pos_orb[4]+lattice_vector([-1,-1, 0]))
                  )), 3, 4, [-1,-1, 0], mode='add')

    # 1st Pt - N real
    model.set_hop(-hop_first_Pt_N, 0, 2, [ 0, 0, 0])
    model.set_hop(-hop_first_Pt_N, 0, 3, [ 1, 0, 0])
    model.set_hop(-hop_first_Pt_N, 0, 4, [ 0, 0, 0])
    model.set_hop(-hop_first_Pt_N, 1, 2, [ 0, 1, 0])
    model.set_hop(-hop_first_Pt_N, 1, 3, [ 0, 0, 0])
    model.set_hop(-hop_first_Pt_N, 1, 4, [ 0, 0, 0])

    # 2nd Pt - N real
    model.set_hop(-hop_second_Pt_N, 0, 2, [ 0, 1, 0])
    model.set_hop(-hop_second_Pt_N, 0, 2, [ 1, 1, 0])
    model.set_hop(-hop_second_Pt_N, 0, 3, [ 0, 0, 0])
    model.set_hop(-hop_second_Pt_N, 0, 3, [ 0,-1, 0])
    model.set_hop(-hop_second_Pt_N, 0, 4, [ 1, 0, 0])
    model.set_hop(-hop_second_Pt_N, 0, 4, [ 0,-1, 0])
    model.set_hop(-hop_second_Pt_N, 1, 2, [ 0, 0, 0])
    model.set_hop(-hop_second_Pt_N, 1, 2, [-1, 0, 0])
    model.set_hop(-hop_second_Pt_N, 1, 3, [ 1, 0, 0])
    model.set_hop(-hop_second_Pt_N, 1, 3, [ 1, 1, 0])
    model.set_hop(-hop_second_Pt_N, 1, 4, [ 0, 1, 0])
    model.set_hop(-hop_second_Pt_N, 1, 4, [-1, 0, 0])

    # 2nd Pt - N imaginary
    model.set_hop(hop_second_Pt_N_soc
                  * array_matrix_Pauli(productVectorTwoNorm(
                      pos_orb[0]-pos_orb[1],
                      pos_orb[1]-(pos_orb[2]+lattice_vector([ 0, 1, 0]))
                  )), 0, 2, [ 0, 1, 0], mode='add')
    model.set_hop(hop_second_Pt_N_soc
                  * array_matrix_Pauli(productVectorTwoNorm(
                      pos_orb[0]-(pos_orb[1]+lattice_vector([ 1, 0, 0])),
                      (pos_orb[1]+lattice_vector([ 1, 0, 0])) -
                      (pos_orb[2]+lattice_vector([ 1, 1, 0]))
                  )), 0, 2, [ 1, 1, 0], mode='add')
    model.set_hop(hop_second_Pt_N_soc
                  * array_matrix_Pauli(productVectorTwoNorm(
                      pos_orb[0]-(pos_orb[1]+lattice_vector([ 0, -1, 0])),
                      (pos_orb[1]+lattice_vector([ 0, -1, 0])) -
                      (pos_orb[3]+lattice_vector([ 0, -1, 0]))
                  )), 0, 3, [ 0, -1, 0], mode='add')
    model.set_hop(hop_second_Pt_N_soc
                  * array_matrix_Pauli(productVectorTwoNorm(
                      pos_orb[0]-(pos_orb[1]+lattice_vector([ 0, -1, 0])),
                      (pos_orb[1]+lattice_vector([ 0,-1, 0])) -
                      (pos_orb[4]+lattice_vector([ 0,-1, 0]))
                  )), 0, 4, [ 0,-1, 0], mode='add')
    model.set_hop(hop_second_Pt_N_soc
                  * array_matrix_Pauli(productVectorTwoNorm(
                      pos_orb[1]-pos_orb[0],
                      pos_orb[0]-pos_orb[2]
                  )), 1, 2, [ 0, 0, 0], mode='add')
    model.set_hop(hop_second_Pt_N_soc
                  * array_matrix_Pauli(productVectorTwoNorm(
                      pos_orb[1]-(pos_orb[0]+lattice_vector([-1, 0, 0])),
                      (pos_orb[0]+lattice_vector([-1, 0, 0])) -
                      (pos_orb[2]+lattice_vector([-1, 0, 0]))
                  )), 1, 2, [-1, 0, 0], mode='add')
    model.set_hop(hop_second_Pt_N_soc
                  * array_matrix_Pauli(productVectorTwoNorm(
                      pos_orb[1]-(pos_orb[0]+lattice_vector([ 0, 1, 0])),
                      (pos_orb[0]+lattice_vector([ 0, 1, 0])) -
                      (pos_orb[3]+lattice_vector([ 1, 1, 0]))
                  )), 1, 3, [ 1, 1, 0], mode='add')
    model.set_hop(hop_second_Pt_N_soc
                  * array_matrix_Pauli(productVectorTwoNorm(
                      pos_orb[1]-(pos_orb[0]+lattice_vector([ 0, 1, 0])),
                      (pos_orb[0]+lattice_vector([ 0, 1, 0])) -
                      (pos_orb[4]+lattice_vector([ 0, 1, 0]))
                  )), 1, 4, [ 0, 1, 0], mode='add')
    model.set_hop(hop_second_Pt_N_soc
                  * array_matrix_Pauli(productVectorTwoNorm(
                      pos_orb[0]-pos_orb[1],
                      pos_orb[1]-pos_orb[3]
                  )), 0, 3, [ 0, 0, 0], mode='add')
    model.set_hop(hop_second_Pt_N_soc
                  * array_matrix_Pauli(productVectorTwoNorm(
                      pos_orb[1]-pos_orb[0],
                      pos_orb[0]-(pos_orb[3]+lattice_vector([ 1, 0, 0]))
                  )), 1, 3, [1, 0, 0], mode='add')
    model.set_hop(hop_second_Pt_N_soc
                  * array_matrix_Pauli(productVectorTwoNorm(
                      pos_orb[0]-(pos_orb[1]+lattice_vector([ 1, 0, 0])),
                      (pos_orb[1]+lattice_vector([ 1, 0, 0])) -
                      (pos_orb[4]+lattice_vector([ 1, 0, 0]))
                  )), 0, 4, [ 1, 0, 0], mode='add')
    model.set_hop(hop_second_Pt_N_soc
                  * array_matrix_Pauli(productVectorTwoNorm(
                      pos_orb[1]-(pos_orb[0]+lattice_vector([-1, 0, 0])),
                      (pos_orb[0]+lattice_vector([-1, 0, 0])) -
                      (pos_orb[4]+lattice_vector([-1, 0, 0]))
                  )), 1, 4, [-1, 0, 0], mode='add')
    return model
