import os

import warnings
import logging
from rdkit import RDLogger

import datamol as dm
from rdkit import Chem
from rdkit.Chem import AllChem
from pymol import cmd
from openbabel import pybel, openbabel
from openeye import oechem

def pybel_converter(input, input_format, output, output_format):
    mols = list(pybel.readfile(input_format, input))
    out = pybel.Outputfile(output_format, output, overwrite=True)

    for mol in mols:
        mol.OBMol.PerceiveBondOrders()
        mol.OBMol.AssignSpinMultiplicity(True)
        mol.OBMol.FindRingAtomsAndBonds()

        # Compute Gasteiger charges
        obChargeModel = openbabel.OBChargeModel.FindType("gasteiger")
        obChargeModel.ComputeCharges(mol.OBMol)

        # Add charges to pybel molecule
        for atom in mol:
            atom.OBAtom.SetPartialCharge(atom.OBAtom.GetPartialCharge())

        out.write(mol)
    out.close()

def pybel_flagger(mol):
    mol.OBMol.PerceiveBondOrders()
    mol.OBMol.AssignSpinMultiplicity(True)
    mol.OBMol.FindRingAtomsAndBonds()

    # Compute Gasteiger charges
    obChargeModel = openbabel.OBChargeModel.FindType("gasteiger")
    obChargeModel.ComputeCharges(mol.OBMol)

    # Add charges to pybel molecule
    for atom in mol:
        atom.OBAtom.SetPartialCharge(atom.OBAtom.GetPartialCharge())
    
    return mol
    

# def pybel_converter(input, input_format, output, output_format):
#     mols = list(pybel.readfile(input_format, input))
#     out = pybel.Outputfile(output_format, output, overwrite=True)
#
#     for mol in mols:
#         mol.OBMol.PerceiveBondOrders()
#         mol.OBMol.AssignSpinMultiplicity(True)
#         mol.OBMol.FindRingAtomsAndBonds()
#         out.write(mol)
#     out.close()

# def pybel_converter(input, input_format, output, output_format):
#     ifs = oechem.oemolistream()
#     ofs = oechem.oemolostream()
#
#     if input_format == "mol2":
#         ifs.SetFormat(oechem.OEFormat_MOL2)
#     elif input_format == "sdf":
#         ifs.SetFormat(oechem.OEFormat_SDF)
#
#     if output_format == "mol2":
#         ofs.SetFormat(oechem.OEFormat_MOL2)
#     elif output_format == "sdf":
#         ofs.SetFormat(oechem.OEFormat_SDF)
#
#     if ifs.open(input):
#         if ofs.open(output):
#             for mol in ifs.GetOEGraphMols():
#                 oechem.OEClearAromaticFlags(mol)
#                 oechem.OEDetermineConnectivity(mol)
#                 oechem.OEFindRingAtomsAndBonds(mol)
#                 oechem.OEPerceiveBondOrders(mol)
#                 oechem.OEAssignImplicitHydrogens(mol)
#                 oechem.OEAssignFormalCharges(mol)
#
#                 oechem.OEAssignAromaticFlags(mol)
#                 for bond in mol.GetBonds():
#                     if bond.IsAromatic():
#                         bond.SetIntType(5)
#                     elif bond.GetOrder() != 0:
#                         bond.SetIntType(bond.GetOrder())
#                     else:
#                         bond.SetIntType(1)
#
#                 # Kekulize the molecule
#                 oechem.OEKekulize(mol)
#
#                 # Add explicit hydrogens
#                 oechem.OEAddExplicitHydrogens(mol)
#
#                 # Assign partial charges
#                 oechem.OEGasteigerPartialCharges(mol)
#
#                 oechem.OEWriteMolecule(ofs, mol)

def getbox(selection='sele', extending = 6.0, software='vina'):
    
    ([minX, minY, minZ],[maxX, maxY, maxZ]) = cmd.get_extent(selection)

    minX = minX - float(extending)
    minY = minY - float(extending)
    minZ = minZ - float(extending)
    maxX = maxX + float(extending)
    maxY = maxY + float(extending)
    maxZ = maxZ + float(extending)
    
    SizeX = maxX - minX
    SizeY = maxY - minY
    SizeZ = maxZ - minZ
    CenterX =  (maxX + minX)/2
    CenterY =  (maxY + minY)/2
    CenterZ =  (maxZ + minZ)/2
    
    cmd.delete('all')
    
    if software == 'vina':
        return {'center_x':CenterX,'center_y': CenterY, 'center_z': CenterZ},{'size_x':SizeX,'size_y': SizeY,'size_z': SizeZ}
    elif software == 'ledock':
        return {'minX':minX, 'maxX': maxX},{'minY':minY, 'maxY':maxY}, {'minZ':minZ,'maxZ':maxZ}
    elif software == 'both':
        return ({'center_x':CenterX,'center_y': CenterY, 'center_z': CenterZ},{'size_x':SizeX,'size_y': SizeY,'size_z': SizeZ}),({'minX':minX, 'maxX': maxX},{'minY':minY, 'maxY':maxY}, {'minZ':minZ,'maxZ':maxZ})
    
    else:
        print('software options must be "vina", "ledock" or "both"')

def generate_ledock_file(receptor='pro.pdb',rmsd=1.0,x=[0,0],y=[0,0],z=[0,0], n_poses=10, l_list=[],l_list_outfile='',out='dock.in'):
    rmsd=str(rmsd)
    x=[str(x) for x in x]
    y=[str(y) for y in y]
    z=[str(z) for z in z]
    n_poses=str(n_poses)

    with open(l_list_outfile,'w') as l_out:
        for element in l_list:
            l_out.write(element)
    l_out.close()

    file=[
        'Receptor\n',
        receptor + '\n\n',
        'RMSD\n',
        rmsd +'\n\n',
        'Binding pocket\n',
        x[0],' ',x[1],'\n',
        y[0],' ',y[1],'\n',
        z[0],' ',z[1],'\n\n',
        'Number of binding poses\n',
        n_poses + '\n\n',
        'Ligands list\n',
        os.path.basename(l_list_outfile) + '\n\n',
        'END']
    
    with open(out,'w') as output:
        for line in file:
            output.write(line)
    output.close()

def dok_to_sdf(dok_path, template_mol2_path, output_sdf_path):
    def parse_dok(file_path):
        with open(file_path) as file:
            lines = file.readlines()
        
        molecules_info = []  # This will store tuples of (coords, score)
        coords = []
        score = None
        for line in lines:
            if line.startswith('REMARK') and 'Score:' in line:
                # Extract score from the REMARK line
                parts = line.split()
                score = parts[-2]  # Assuming the score is always the second last element
            elif line.startswith('ATOM'):
                parts = line.split()
                atom_index = int(parts[1])
                x, y, z = map(float, parts[5:8])
                coords.append((atom_index, x, y, z))
            elif line.startswith('END'):
                if coords:
                    molecules_info.append((coords, score))
                    coords = []
                    score = None
        if coords:  # If there are coords left after parsing
            molecules_info.append((coords, score))
        
        return molecules_info

    def update_and_write_molecules(molecules_info, template_path, output_path):
        output = pybel.Outputfile('sdf', output_path, overwrite=True)
        for coords, score in molecules_info:
            # Re-read the template molecule for each set of coordinates
            template_mol = next(pybel.readfile('mol2', template_path))
            for atom_index, x, y, z in coords:
                if atom_index - 1 < len(template_mol.atoms):
                    atom = template_mol.atoms[atom_index - 1]
                    atom.OBAtom.SetVector(x, y, z)
            # Add score as a property
            if score is not None:
                template_mol.data['Score'] = score
            output.write(template_mol)
        output.close()

    molecules_info = parse_dok(dok_path)
    update_and_write_molecules(molecules_info, template_mol2_path, output_sdf_path)

def unzip_mol2(in_mol2, save_dir):
    save_path_list = []
    for mol in pybel.readfile("mol2", in_mol2):
        # Fix the title
        name = mol.title

        #IO
        save_path = os.path.join(save_dir, name + ".mol2")
        out_class = pybel.Outputfile("mol2", save_path, overwrite=True)

        # Operation on the MOL
        mol.addh()
        # charges = mol.calccharges(model="gasteiger")

        # IO
        out_class.write(mol)
        save_path_list.append(os.path.abspath(save_path))
    
    return save_path_list

def unzip_sdf(in_sdf, save_dir):
    save_path_list = []
    for mol in dm.read_sdf(in_sdf):
        name = mol.GetProp("_Name")
        save_path = os.path.join(save_dir, name + ".sdf")
        
        mol_h = dm.add_hs(mol, add_coords=True)
        # AllChem.Compute2DCoords(mol)
        # AllChem.EmbedMolecule(mol)
        
        dm.to_sdf([mol_h], save_path)
        
        save_path_list.append(os.path.abspath(save_path))
    
    return save_path_list

def warning_suppresser(level="BADASS"):
    if not level in ["STRONG", "BADASS"]:
        raise ValueError("Invalid level. Please choose from 'STRONG' or 'BADASS'.")

    if level == "BADASS":
        class SuppressLoggingFilter(logging.Filter):
            def filter(self, record):
                return False

        # Apply the filter to the root logger
        for handler in logging.root.handlers:
            handler.addFilter(SuppressLoggingFilter())

        print("Confirming level: BADASS")
    else:
        print("Confirming level: STRONG")

    # Disable general warnings
    logging.getLogger().setLevel(logging.CRITICAL)
    warnings.filterwarnings('ignore')
    warnings.filterwarnings('ignore', module="pandas.*")
    warnings.filterwarnings('ignore', module="numpy.*")
    warnings.filterwarnings('ignore', module="sklearn.*")
    warnings.filterwarnings('ignore', module="rdkit.*")
    warnings.filterwarnings('ignore', module="datamol.*")

    # Disable Datamol/Rdkit, Openbabel
    RDLogger.logger().setLevel(RDLogger.CRITICAL)
    pybel.ob.obErrorLog.StopLogging()
