import numpy as np

def productVectorFour(orb12,orb23,orb34,orb45):
   
    return np.cross(orb12, orb23) + np.cross(orb34,orb45)

def productVectorFourNorm(orb12,orb23,orb34,orb45):
    vector = productVectorFour(orb12,orb23,orb34,orb45)
   
    return vector/np.linalg.norm(vector)
    
def productVectorTwo(orb12,orb23):
   
    return np.cross(orb12,orb23)

def productVectorTwoNorm(orb12,orb23):
    vector = productVectorTwo(orb12,orb23)
   
    return vector/np.linalg.norm(vector)
    
def array_matrix_Pauli(vetor):

    sigma_x = np.array([ 0, 1, 0, 0])
    sigma_y = np.array([ 0, 0, 1, 0])
    sigma_z = np.array([ 0, 0, 0, 1])
    
    return vetor[0]*sigma_x + vetor[1]*sigma_y + vetor[2]*sigma_z