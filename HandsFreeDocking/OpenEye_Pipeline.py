import os
from typing import List, Tuple, Optional
import logging
from pathlib import Path
from tempfile import gettempdir

import concurrent
from concurrent.futures import ThreadPoolExecutor, as_completed
from joblib import Parallel, delayed
from tqdm import tqdm

import biotite.structure.io.pdb as pdb
import biotite.database.rcsb as rcsb
import biotite.structure.io as strucio
import biotite.structure as struc

from pymol import cmd

from openeye import oechem
from openeye import oeomega
from openeye import oegrid
from openeye import oespruce
from openeye import oedocking
from openeye import oeshape

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)


def make_design(infile: str, savepath: str, mtzfile: Optional[str] = None, loopdbfile: Optional[str] = None):
    if not os.path.exists(infile):
        raise ValueError(f"Unable to open {infile} for reading")

    include_loop = loopdbfile is not None
    include_ed = mtzfile is not None and os.path.exists(mtzfile)
    ed = oegrid.OESkewGrid()

    if include_ed:
        if not oegrid.OEReadMTZ(mtzfile, ed, oegrid.OEMTZMapType_Fwt):
            raise ValueError(f"Unable to read electron density file {mtzfile}")

    ifs = oechem.oemolistream()
    if not ifs.open(infile):
        raise ValueError(f"Unable to open {infile} for reading")

    if ifs.GetFormat() not in [oechem.OEFormat_PDB, oechem.OEFormat_CIF]:
        raise ValueError("Only works for .pdb or .cif input files")

    ifs.SetFlavor(oechem.OEFormat_PDB,
                  oechem.OEIFlavor_PDB_Default | oechem.OEIFlavor_PDB_DATA | oechem.OEIFlavor_PDB_ALTLOC)

    mol = oechem.OEGraphMol()
    if not oechem.OEReadMolecule(ifs, mol):
        raise ValueError(f"Unable to read molecule from {infile}")

    allow_filter_errors = False
    metadata = oespruce.OEStructureMetadata()
    filter_opts = oespruce.OESpruceFilterOptions()
    makedu_opts = oespruce.OEMakeDesignUnitOptions()
    makedu_opts.GetPrepOptions().GetBuildOptions().GetLoopBuilderOptions().SetBuildTails(False)
    if include_loop:
        makedu_opts.GetPrepOptions().GetBuildOptions().GetLoopBuilderOptions().SetLoopDBFilename(loopdbfile)

    filter = oespruce.OESpruceFilter(filter_opts, makedu_opts)
    ret_filter = filter.StandardizeAndFilter(mol, ed, metadata)
    if ret_filter != oespruce.OESpruceFilterIssueCodes_Success:
        oechem.OEThrow.Warning("This structure fails spruce filter due to: ")
        oechem.OEThrow.Warning(filter.GetMessages())
        if not allow_filter_errors:
            raise ValueError("This structure fails spruce filter")

    if include_ed:
        design_units = oespruce.OEMakeDesignUnits(mol, ed, metadata, makedu_opts)
    else:
        design_units = oespruce.OEMakeDesignUnits(mol, metadata, makedu_opts)

    validator = oespruce.OEValidateDesignUnit()

    base_name = os.path.splitext(os.path.basename(infile))[0] + "_DU_{}.oedu"
    for i, design_unit in enumerate(design_units):
        ret_validator = validator.Validate(design_unit, metadata)

        if ret_validator != oespruce.OESpruceFilterIssueCodes_Success:
            oechem.OEThrow.Warning("This generated DU did not pass DU validator.")
            oechem.OEThrow.Warning(validator.GetMessages())

        saving = os.path.join(savepath, base_name.format(i))

        oechem.OEWriteDesignUnit(saving, design_unit)


def make_receptor(input_oedesign: str, output_oedesign: str, crystal_sdf: str) -> str:
    # Load the Crystal structure
    ifs_crystal = oechem.oemolistream()
    if not ifs_crystal.open(crystal_sdf):
        logging.error(f"Unable to open {crystal_sdf} for reading")

    crystal_mol = list(ifs_crystal.GetOEGraphMols())[0]

    # IO Settings
    ifs = oechem.oeifstream()
    ifs.open(input_oedesign)

    ofs = oechem.oeofstream()
    ofs.open(output_oedesign)

    # Configure settings
    recOpts = oedocking.OEMakeReceptorOptions()
    recOpts.SetBoxMol(crystal_mol)
    recOpts.SetBoxExtension(2.00)
    recOpts.SetNegativeImageType(2)
    recOpts.SetTargetMask(oechem.OEDesignUnitComponents_TargetComplexNoSolvent)

    # Make Receptor
    du = oechem.OEDesignUnit()
    while oechem.OEReadDesignUnit(ifs, du):
        if oedocking.OEMakeReceptor(du, recOpts):
            oechem.OEWriteDesignUnit(ofs, du)
        else:
            oechem.OEThrow.Warning("%s: %s" % (du.GetTitle(), "Failed to make receptor"))

    # Close the files
    ifs.close()
    ofs.close()

    return output_oedesign


class OpenEye_Docking:
    def __init__(self, workdir: Path, pdb_ID: Path, mtz: Optional[Path], crystal_path: Path,
                 docking_tuple: List[Tuple[str, str]]):
        """
        :param workdir: Path to the working directory
        :param pdb_ID: Path to the PDB file
        :param mtz: Optional Path to the MTZ file
        :param crystal_path: Path to the crystal structure file
        :param docking_tuple: List of tuples containing SMILES strings and names
        """
        # Setup working directory
        self.workdir: Path = workdir
        self.workdir.mkdir(parents=True, exist_ok=True)

        # Set up the output directory
        self.output_dir = self.workdir / "output"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Get the current working directory
        self.starting_dir = os.getcwd()

        # Setup macromolecule variables
        self.pdb_ID = pdb_ID

        self.mtz = mtz.resolve() if mtz else None

        # Setup input variable
        self.crystal_path = crystal_path.resolve()
        self.docking_tuple: List[Tuple[str, str]] = docking_tuple

    def _source_macro(self) -> Path:
        cmd.reinitialize()
        cmd.load(str(self.pdb_ID), "protein_XX")
        cmd.load(str(self.crystal_path), "crystal")
        cmd.create("complex", "protein_XX or crystal")

        save_path = self.workdir / "complex.pdb"
        cmd.save(str(save_path), "complex")

        return save_path

    @staticmethod
    def transform_smile(smile: str) -> oechem.OEMol:
        mol_openeye = oechem.OEMol()
        oechem.OESmilesToMol(mol_openeye, smile)
        return mol_openeye

    @staticmethod
    def conf_fit(reference_sdf: str, probe_sdf: str, output_sdf: str):
        # Load Reference
        rfs = oechem.oemolistream()
        if not rfs.open(reference_sdf):
            raise FileNotFoundError(f"Unable to open {reference_sdf} for reading")

        # Load the Probe
        ifs = oechem.oemolistream()
        if not ifs.open(probe_sdf):
            raise FileNotFoundError(f"Unable to open {probe_sdf} for reading")

        # Setup the output
        ofs = oechem.oemolostream()
        if not ofs.open(output_sdf):
            raise FileNotFoundError(f"Unable to open {output_sdf} for writing")

        refmol = oechem.OEMol()
        oechem.OEReadMolecule(rfs, refmol)

        overlayOpts = oeshape.OEFlexiOverlayOptions()
        overlay = oeshape.OEFlexiOverlay(overlayOpts)
        overlay.SetupRef(refmol)

        for fitmol in ifs.GetOEMols():
            results = overlay.Overlay(fitmol)
            for res, conf in zip(results, fitmol.GetConfs()):
                print("Fit Title: %-4s Tanimoto Combo: %.2f Energy: %2f"
                      % (fitmol.GetTitle(), res.GetTanimotoCombo(), res.GetInternalEnergy()))
                oechem.OESetSDData(conf, "Tanimoto Combo", "%.2f" % res.GetTanimotoCombo())
                oechem.OESetSDData(conf, "Energy", "%.2f" % res.GetInternalEnergy())

            oechem.OEWriteMolecule(ofs, fitmol)

    @staticmethod
    def generate_confs(mol: oechem.OEMol, max_confs: int):
        """
        Generate conformers for a given molecule using OpenEye Omega.

        Args:
            mol (oechem.OEMol): The molecule for which conformers need to be generated.
            max_confs (int): The maximum number of conformers to generate.

        Returns:
            None
        """
        rms = 0.5
        strict_stereo = False

        # OMEGA settings
        omegaOpts = oeomega.OEOmegaOptions(oeomega.OEOmegaSampling_Dense)
        omegaOpts.SetMaxConfs(max_confs)
        omegaOpts.SetStrictStereo(strict_stereo)
        omegaOpts.SetRMSThreshold(rms)

        omega = oeomega.OEOmega(omegaOpts)
        error_level = oechem.OEThrow.GetLevel()

        # Turn off OEChem warnings
        oechem.OEThrow.SetLevel(oechem.OEErrorLevel_Error)

        status = omega(mol)
        ret_code = omega.Build(mol)

        # Turn OEChem warnings back on
        oechem.OEThrow.SetLevel(error_level)

    @staticmethod
    def run_oedocking(oemol: oechem.OEMol, design_final: str, output_sdf: str, max_confs: int):
        # IO settings
        rfs = oechem.oeifstream()
        if not rfs.open(design_final):
            oechem.OEThrow.Fatal("Error")

        ofs = oechem.oemolostream()
        if not ofs.open(output_sdf):
            oechem.OEThrow.Fatal("Error")

        du = oechem.OEDesignUnit()
        if not oechem.OEReadDesignUnit(rfs, du):
            oechem.OEThrow.Fatal("Failed to read design unit")

        # DEBUG
        if not du.HasReceptor():
            logging.error(f"Design unit {du.GetTitle()} does not contain a receptor")
            raise Exception(f"Design unit {du.GetTitle()} does not contain a receptor")

        # Docking options
        dockOpts = oedocking.OEDockOptions()
        dockOpts.SetScoreMethod(oedocking.OEScoreType_PLP)
        dockOpts.SetResolution(oedocking.OESearchResolution_High)

        dock = oedocking.OEDock(dockOpts)
        dock.Initialize(du)

        dockedMol = oechem.OEMol()
        retCode = dock.DockMultiConformerMolecule(dockedMol, oemol, max_confs)

        if retCode != oedocking.OEDockingReturnCode_Success:
            log_error = oedocking.OEDockingReturnCodeGetName(retCode)
            ligand_failed = oemol.GetTitle()
            raise Exception(f"Internal error {log_error}")
        else:
            scores = []
            for i, docked_conf in enumerate(dockedMol.GetConfs()):
                # Add name to SDF
                ID = os.path.splitext(os.path.basename(output_sdf))[0]
                docked_conf.SetTitle(f"{ID}-P{i}")

                # Set the score
                sdtag = oedocking.OEDockMethodGetName(dockOpts.GetScoreMethod())
                oedocking.OESetSDScore(docked_conf, dock, sdtag)
                dock.AnnotatePose(docked_conf)

                score = oechem.OEGetSDData(docked_conf, sdtag)
                scores.append(float(score))

            oechem.OEWriteMolecule(ofs, dockedMol)

        rfs.close()
        ofs.close()

    @staticmethod
    def run_posit_docking(oemol: oechem.OEMol, design_final: str, output_sdf: str, max_confs: int):
        class MyOptions(oedocking.OEPositOptions):
            def __init__(self):
                super().__init__()
                param1 = oechem.OEUIntParameter("-numPoses", max_confs)
                param1.SetBrief("Number of poses to generate")
                self._param1 = self.AddParameter(param1)

            def GetNumPoses(self) -> int:
                if self._param1.GetHasValue():
                    return int(self._param1.GetStringValue())
                return int(self._param1.GetStringDefault())

        # Load options
        positOpts = MyOptions()

        # IO settings
        rfs = oechem.oeifstream()
        rfs.open(design_final)

        ofs = oechem.oemolostream()
        ofs.open(output_sdf)
        ID = os.path.splitext(os.path.basename(output_sdf))[0]

        poser = oedocking.OEPosit()
        du = oechem.OEDesignUnit()
        count = 0
        while oechem.OEReadDesignUnit(rfs, du):
            if not du.HasReceptor():
                oechem.OEThrow.Fatal(f"Design unit {du.GetTitle()} does not contain a receptor")
            poser.AddReceptor(du)
            count += 1
        if count == 0:
            oechem.OEThrow.Fatal("Receptor input does not contain any design unit")

        results = oedocking.OEPositResults()
        ret_code = poser.Dock(results, oemol, positOpts.GetNumPoses())

        for i, result in enumerate(results.GetSinglePoseResults()):
            poseDU: oechem.OEMolBase = result.GetPose()
            score = str(result.GetProbability())

            oechem.OESetSDData(poseDU, "Probability", score)
            poseDU.SetTitle(f"{ID}-P{i}")

            oechem.OEWriteMolecule(ofs, oechem.OEGraphMol(poseDU))

    def run_oedocking_pipeline(self, n_cpu: int = 4, confs: int = 100, mtz: Optional[Path] = None, mode: str = "oe"):
        assert mode in ["oe", "posit"], "Mode must be either 'oe' or 'posit'"

        protein_path = self._source_macro()
        protein_basename = protein_path.name

        # Make design unit
        logging.info("Making Design")

        tempdir = Path(gettempdir())
        design_unit: Path = tempdir / "complex_DU_0.oedu"
        receptor_unit: Path = self.workdir / f"{self.pdb_ID.stem}_Receptor.oedu"

        make_design(
            infile=str(protein_path),
            savepath=str(tempdir),
            mtzfile=str(mtz) if mtz else None
        )
        make_receptor(
            input_oedesign=str(design_unit),
            output_oedesign=str(receptor_unit),
            crystal_sdf=str(self.crystal_path)
        )

        logging.info("Receptor Made")

        def runner(docking_info: Tuple[str, str]):
            smile, name = docking_info

            # Transform and generate conformers
            mol_openeye = self.transform_smile(smile)
            self.generate_confs(mol_openeye, max_confs=confs)

            try:
                outname_sdf: Path = self.output_dir / f"{name}_Eye.sdf"
                if mode == "oe":
                    self.run_oedocking(
                        oemol=mol_openeye,
                        design_final=str(receptor_unit),
                        output_sdf=str(outname_sdf),
                        max_confs=confs
                    )
                elif mode == "posit":
                    self.run_posit_docking(
                        oemol=mol_openeye,
                        design_final=str(receptor_unit),
                        output_sdf=str(outname_sdf),
                        max_confs=confs
                    )
            except Exception as e:
                logging.error(e)
                logging.error(f"Failed to dock {name}")

        with ThreadPoolExecutor(max_workers=n_cpu) as executor:
            futures = [executor.submit(runner, exe_tup) for exe_tup in self.docking_tuple]
            for future in tqdm(as_completed(futures), total=len(futures)):
                try:
                    future.result()
                except concurrent.futures.BrokenExecutor:
                    logging.error("Internal Error")
                except Exception as e:
                    logging.error(f"Failed to dock with error {e}")