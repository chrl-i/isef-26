import numpy as np
import os
from Bio.PDB import PDBParser
from collections import defaultdict

#si-derived moieties (magnetic susceptibility in 10^-6 cm³/mol, CGS units)
MOIETY_TENSORS = {
    "peptide": np.diag([-40.302, -40.302, -40.302 + -5.361]),
    "alkane": np.diag([-11.645, -11.645, -11.645 + -1.633]),
    "benzene": np.diag([-55.307, -55.307, -55.307 + -63.406]),
    "phenol": np.diag([-60.902, -60.902, -60.902 + -63.408]),
    "indole": np.diag([-85.002, -85.002, -85.002 + -86.088]),
    "imidazole": np.diag([-43.052, -43.052, -43.052 + -24.41]),
    "formamide": np.diag([-22.97, -22.97, -22.97 + -9.00]),
    "formic_acid": np.diag([-19.20, -19.20, -19.20 + -9.43]),
    "dimethyl_sulfide": np.diag([-44.92, -44.92, -44.92 + -3.59]),
    "methanol": np.diag([-21.372, -21.372, -21.372 + -1.00]),
    "methanethiol": np.diag([-35.262, -35.262, -35.262 + -1.00]),
    "methylamine": np.diag([-11.645, -11.645, -11.645 + -1.633])
}

#amino residue assignment
RESIDUE_TO_MOIETY = defaultdict(lambda: ["peptide"])
RESIDUE_TO_MOIETY.update({
    "ALA": ["peptide", "alkane"],
    "VAL": ["peptide", "alkane"],
    "LEU": ["peptide", "alkane"],
    "ILE": ["peptide", "alkane"],
    "PHE": ["peptide", "benzene"],
    "TYR": ["peptide", "phenol"],
    "TRP": ["peptide", "indole"],
    "HIS": ["peptide", "imidazole"],
    "ASN": ["peptide", "formamide"],
    "GLN": ["peptide", "formamide"],
    "ASP": ["peptide", "formic_acid"],
    "GLU": ["peptide", "formic_acid"],
    "MET": ["peptide", "dimethyl_sulfide"],
    "SER": ["peptide", "methanol"],
    "THR": ["peptide", "methanol"],
    "CYS": ["peptide", "methanethiol"],
    "PRO": ["peptide", "methylamine"],
    "LYS": ["peptide", "methylamine"],
    "ARG": ["peptide", "methylamine"]
})

def get_structure(pdb_filename):
    parser = PDBParser(QUIET=True)
    return parser.get_structure("MP", pdb_filename)

#local rotation matrix from 3 atoms
def get_rotation_matrix(coords):
    x_axis = coords[1] - coords[0]
    x_axis /= np.linalg.norm(x_axis)
    temp = coords[2] - coords[0]
    z_axis = np.cross(x_axis, temp)
    z_axis /= np.linalg.norm(z_axis)
    y_axis = np.cross(z_axis, x_axis)
    return np.vstack([x_axis, y_axis, z_axis]).T

#rotate local tensor into global coordinates
def transform_tensor(K_local, Q):
    return Q @ K_local @ Q.T

#main comp function
def compute_total_tensor(pdb_filename):
    structure = get_structure(pdb_filename)
    total_tensor = np.zeros((3, 3))
    count = 0
    
    for model in structure:
        for chain in model:
            for residue in chain:
                resname = residue.get_resname()
                atoms = list(residue.get_atoms())
                
                if len(atoms) < 3:
                    continue  #need at least 3 atoms to define local frame
                
                try:
                    coords = np.array([atoms[0].coord, atoms[1].coord, atoms[2].coord])
                    Q = get_rotation_matrix(coords)
                except Exception:
                    continue  #skip ill-defined geometries
                
                for moiety in RESIDUE_TO_MOIETY[resname]:
                    K_local = MOIETY_TENSORS[moiety]
                    K_global = transform_tensor(K_local, Q)
                    total_tensor += K_global
                    count += 1
    
    print(f"Processed {count} moieties.")
    return total_tensor

#compute anisotropy from tensor
def compute_anisotropy(tensor):
    eigvals = np.linalg.eigvalsh(tensor)
    anisotropy = max(eigvals) - min(eigvals)
    return eigvals, anisotropy

#unit conversion
def convert_to_si(tensor_cgs):
    """Convert from CGS (10^-6 cm³/mol) to SI (m³/mol)"""
    #1 cm³ = 10^-6 m³
    #so 10^-6 cm³/mol = 10^-6 × 10^-6 m³/mol = 10^-12 m³/mol
    return tensor_cgs * 1e-12

def convert_cgs_to_dimensionless(tensor_cgs):
    """Convert CGS susceptibility to dimensionless SI susceptibility"""
    #in CGS: χ has units of cm³/mol
    #in SI: χ is dimensionless
    #conversion factor: multiply by (μ₀/4π) × (density/molar_mass)
    #for proteins, approximate conversion factor is ~1.33e-6
    return tensor_cgs * 1.33e-6

#main exe
if __name__ == "__main__":
    #print("Current directory:", os.getcwd())
    pdb_file = "1igt.pdb"  #hereeeeeeeeeeeeeeliestheimportttttttttttttttt
    #pdb_file = "1hho.pdb"  #x, cause:prosthetic group
    #pdb_file = "1dpx.pdb"  #9% off
    #pdb_file = "3v03.pdb"  #8.9% off 

    # Compute tensor
    total_tensor = compute_total_tensor(pdb_file)
    eigvals, delta_chi = compute_anisotropy(total_tensor)
    
    print("\n" + "="*60)
    print("MAGNETIC SUSCEPTIBILITY RESULTS")
    print("="*60)
    
    print(f"\nTotal Susceptibility Tensor (CGS units: 10⁻⁶ cm³/mol):")
    print(total_tensor)
    
    print(f"\nEigenvalues (CGS units: 10⁻⁶ cm³/mol):")
    print(eigvals)
    
    print(f"\nDiamagnetic Anisotropy Δχ (CGS units: 10⁻⁶ cm³/mol):")
    print(f"{delta_chi:.2f}")
    
    #convert to SI units
    tensor_si = convert_to_si(total_tensor)
    eigvals_si = convert_to_si(eigvals)
    delta_chi_si = convert_to_si(delta_chi)
    
    print(f"\n" + "-"*40)
    print("CONVERTED TO SI UNITS:")
    print("-"*40)
    
    print(f"\nTotal Susceptibility Tensor (SI units: m³/mol):")
    print(tensor_si)
    
    print(f"\nEigenvalues (SI units: m³/mol):")
    print(eigvals_si)
    
    print(f"\nDiamagnetic Anisotropy Δχ (SI units: m³/mol):")
    print(f"{delta_chi_si:.2e}")
    
    #convert to dimensionless susceptibility
    tensor_dimensionless = convert_cgs_to_dimensionless(total_tensor)
    eigvals_dimensionless = convert_cgs_to_dimensionless(eigvals)
    delta_chi_dimensionless = convert_cgs_to_dimensionless(delta_chi)
    
    print(f"\n" + "-"*40)
    print("DIMENSIONLESS SUSCEPTIBILITY (SI):")
    print("-"*40)
    
    print(f"\nTotal Susceptibility Tensor (dimensionless):")
    print(tensor_dimensionless)
    
    print(f"\nEigenvalues (dimensionless):")
    print(eigvals_dimensionless)
    
    print(f"\nDiamagnetic Anisotropy Δχ (dimensionless):")
    print(f"{delta_chi_dimensionless:.2e}")
    
    print(f"\n" + "="*60)
    print("UNIT EXPLANATION:")
    print("="*60)
    print("• CGS units (10⁻⁶ cm³/mol): Conv. magnetic susceptibility units")
    print("• SI units (m³/mol): Volume susceptibility in SI (what you need for diamag formula!)")
    print("• Dimensionless: Standard SI magnetic susceptibility (what you see in χ tables)")