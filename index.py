from definitions.band_rectangular_cell import *
from definitions.band_primitive_cell import *
from MNX import *
import sys

if len(sys.argv) != 3:
    print("Use: ./index.sh")
    sys.exit(1)

type_cell = sys.argv[1] # primitive or rectangular
w_soc = bool(sys.argv[2])

if w_soc == False:
    variables_soc = [ 0, 0, 0, 0]

if type_cell == 'rectangular':
    plot_band_rectangular(variables = variables_soc, 
            w_soc = 'soc' if w_soc == True else 'wosoc', 
            nks = 242, 
            jacutingaite = jacutingaite, 
            jacutingaite_name = jacutingaite_name,
            occ = [0, 1, 2, 3]
            )
    
if type_cell == 'primitive':
    plot_band_primitive(variables = variables_soc, 
            w_soc = 'soc' if w_soc == True else 'wosoc', 
            nks = 242, 
            jacutingaite = jacutingaite, 
            jacutingaite_name = jacutingaite_name,
            occ = [0, 1]
            )
