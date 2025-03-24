import logging
from openeye import oechem
from openeye import oeiupac
from openeye import oeomega
from openeye import oequacpac
from typing import List

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',  # Define the log format
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def get_chirality_and_stereo(oemol: oechem.OEGraphMol):
    result = []
    for atom in oemol.GetAtoms():
        chiral = atom.IsChiral()
        stereo = oechem.OEAtomStereo_Undefined
        if atom.HasStereoSpecified(oechem.OEAtomStereo_Tetrahedral):
            v = []
            for nbr in atom.GetAtoms():
                v.append(nbr)
            stereo = atom.GetStereo(v, oechem.OEAtomStereo_Tetrahedral)

        if chiral or stereo != oechem.OEAtomStereo_Undefined:
            chirality = "undefined"
            if stereo == oechem.OEAtomStereo_RightHanded:
                chirality = "R"
            elif stereo == oechem.OEAtomStereo_LeftHanded:
                chirality = "S"
            result.append(f"{atom.GetIdx()}{oechem.OEGetAtomicSymbol(atom.GetAtomicNum())}_{chirality}")
    return "|".join(result)


def from_smiles_to_oemol(smiles) -> oechem.OEMol:
    oemol: oechem.OEMol = oechem.OEMol()
    oechem.OESmilesToMol(oemol, smiles)
    return oemol


def cansmi(smile, isomeric=True, kekule=True):
    """
    Generate a canonical SMILES representation of a oemolecule.

    Args:
        oemol (oechem.OEMol): The oemolecule to generate SMILES for.
        isomeric (bool): Flag indicating whether to include isomeric information in the SMILES.
        kekule (bool): Flag indicating whether to kekulize the oemolecule before generating SMILES.

    Returns:
        str: The canonical SMILES representation of the oemolecule.
    """
    oemol = oechem.OEMol()
    oechem.OESmilesToMol(oemol, smile)

    oechem.OEFindRingAtomsAndBonds(oemol)
    oechem.OEAssignAromaticFlags(oemol, oechem.OEAroModel_OpenEye)
    smiflag = oechem.OESMILESFlag_Canonical
    if isomeric:
        smiflag |= oechem.OESMILESFlag_ISOMERIC

    if kekule:
        for bond in oemol.GetBonds(oechem.OEIsAromaticBond()):
            bond.SetIntType(5)
        oechem.OECanonicalOrderAtoms(oemol)
        oechem.OECanonicalOrderBonds(oemol)
        oechem.OEClearAromaticFlags(oemol)
        oechem.OEKekulize(oemol)

    # Strip Salt
    oechem.OEDeleteEverythingExceptTheFirstLargestComponent(oemol)

    smile = oechem.OECreateSmiString(oemol, smiflag)
    return smile


def mol2name(in_smile):
    """
    Convert a SMILES string to an IUPAC name.

    Args:
        in_smile (str): The input SMILES string.

    Returns:
        str: The IUPAC name of the molecule.

    """
    mol = oechem.OEGraphMol()
    oechem.OESmilesToMol(mol, in_smile)

    language = oeiupac.OEGetIUPACLanguage("american")
    charset = oeiupac.OEGetIUPACCharSet("default")
    style = oeiupac.OEGetIUPACNamStyle("cas")

    if mol.GetDimension() == 3:
        oechem.OEPerceiveChiral(mol)
        oechem.OE3DToAtomStereo(mol)
        oechem.OE3DToBondStereo(mol)

    name = oeiupac.OECreateIUPACName(mol, style)

    if language > 0:
        name = oeiupac.OEToLanguage(name, language)

    if charset == oeiupac.OECharSet_ASCII:
        name = oeiupac.OEToASCII(name)
    elif charset == oeiupac.OECharSet_UTF8:
        name = oeiupac.OEToUTF8(name)
    elif charset == oeiupac.OECharSet_HTML:
        name = oeiupac.OEToHTML(name)
    elif charset == oeiupac.OECharSet_SJIS:
        name = oeiupac.OEToSJIS(name)
    elif charset == oeiupac.OECharSet_EUCJP:
        name = oeiupac.OEToEUCJP(name)
    return name


def fix_3dmol(oemol, addhs, protonate) -> oechem.OEGraphMol:
    # TODO: Check if this is legit
    # Determine the connectivity it most effective when the molecule is highly broken
    oechem.OEDetermineConnectivity(oemol)

    # TODO: Check if this is legit
    oechem.OEFindRingAtomsAndBonds(oemol)

    # TODO: Check if this is legit
    # Try to figure it out the bond order may fucked up the kekulization
    # oechem.OEPerceiveBondOrders(oemol)

    oechem.OEAssignImplicitHydrogens(oemol)
    oechem.OEAssignFormalCharges(oemol)

    oechem.OEClearAromaticFlags(oemol)
    oechem.OEAssignAromaticFlags(oemol)
    for bond in oemol.GetBonds():
        if bond.IsAromatic():
            bond.SetIntType(5)
        elif bond.GetOrder() != 0:
            bond.SetIntType(bond.GetOrder())
        else:
            bond.SetIntType(1)
    oechem.OEKekulize(oemol)

    if protonate == True:
        oequacpac.OEGetReasonableProtomer(oemol)

    if addhs == True:
        oechem.OEAddExplicitHydrogens(oemol)

    return oechem.OEGraphMol(oemol)


def gen_3dmol(smile: str, protonate: bool, gen3d: bool, enum_isomers: bool) -> List[oechem.OEGraphMol]:
    smile_fixed = cansmi(smile, isomeric=True, kekule=True)
    oemol: oechem.OEMol = from_smiles_to_oemol(smile_fixed)

    # Flipping Options
    flipperOpts = oeomega.OEFlipperOptions()

    # Conf Gen initialization
    omega = oeomega.OEOmega()  # For multi confs
    builder = oeomega.OEConformerBuilder()  # For single conf

    if protonate == True:
        logger.info("Protonating the molecule")
        oequacpac.OEGetReasonableProtomer(oemol)

    if enum_isomers == True and gen3d == True:
        enantiomers_3D: List[oechem.OEGraphMol] = []
        for i, enantiomer in enumerate(oeomega.OEFlipper(oemol, flipperOpts)):
            enantiomer = oechem.OEMol(enantiomer)

            logger.info("Generating 3D coordinates")
            ret_code = omega.Build(enantiomer)

            stereo_desc = get_chirality_and_stereo(oechem.OEGraphMol(enantiomer))

            oechem.OESetSDData(enantiomer, "Chiral_ID", f"Stereo_{i}")
            oechem.OESetSDData(enantiomer, "Chiral_Atoms", stereo_desc)
            enantiomers_3D.append(oechem.OEGraphMol(enantiomer))
        
        return enantiomers_3D

    if enum_isomers == False and gen3d == True:
        logger.info("Generating 3D coordinates")
        ret_code = builder.Build(oemol)
        return [oechem.OEGraphMol(oemol)]
