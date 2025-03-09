#!/usr/bin/env python3
"""
Example script for H2 (hydrogen molecule) in both normal and antimatter forms.
"""

import sys
import os
import numpy as np

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the antimatter quantum chemistry package
from antimatter_qchem import AntimatterQuantumChemistry

def run_h2_example():
    """Test both normal matter and antimatter H2 systems."""
    print("=" * 60)
    print("HYDROGEN MOLECULE (H2) EXAMPLE")
    print("=" * 60)
    
    # Create the framework
    print("Initializing Antimatter Quantum Chemistry Framework...")
    aqc = AntimatterQuantumChemistry()
    
    # Bond distance for H2 in Bohr
    bond_distance = 0.74
    
    print("\nCreating H2 systems...")
    # Normal H2
    normal_h2 = aqc.create_antimatter_system(
        name="H2_normal",
        nuclei=[
            ("H", 1, 1.0, np.array([0.0, 0.0, 0.0]), False),
            ("H", 1, 1.0, np.array([0.0, 0.0, bond_distance]), False)
        ],
        n_electrons=2,
        n_positrons=0
    )
    print(f"  Created normal H2 system with bond distance {bond_distance} Bohr")
    
    # Anti-H2
    anti_h2 = aqc.create_antimatter_system(
        name="H2_antimatter",
        nuclei=[
            ("H", 1, -1.0, np.array([0.0, 0.0, 0.0]), True),
            ("H", 1, -1.0, np.array([0.0, 0.0, bond_distance]), True)
        ],
        n_electrons=0,
        n_positrons=2
    )
    print(f"  Created anti-H2 system with bond distance {bond_distance} Bohr")
    
    # Mixed H2 (matter-antimatter combination)
    mixed_h2 = aqc.create_antimatter_system(
        name="H2_mixed",
        nuclei=[
            ("H", 1, 1.0, np.array([0.0, 0.0, 0.0]), False),
            ("H", 1, -1.0, np.array([0.0, 0.0, bond_distance]), True)
        ],
        n_electrons=1,
        n_positrons=1
    )
    print(f"  Created mixed H2 system (H + anti-H) with bond distance {bond_distance} Bohr")
    
    # Solve all systems
    print("\nSolving normal H2...")
    normal_result = aqc.solve_system("H2_normal", method="classical")
    print(f"Normal H2 Energy: {normal_result['energy']:.6f} Hartree")
    
    print("\nSolving anti-H2...")
    anti_result = aqc.solve_system("H2_antimatter", method="classical")
    print(f"Anti-H2 Energy: {anti_result['energy']:.6f} Hartree")
    
    print("\nSolving mixed H2...")
    mixed_result = aqc.solve_system("H2_mixed", method="classical")
    print(f"Mixed H2 Energy: {mixed_result['energy']:.6f} Hartree")
    
    # Compare matter and antimatter
    print("\nComparing matter and antimatter H2...")
    comparison = aqc.compare_matter_antimatter("H2_normal")
    print(f"Energy Difference: {comparison['energy_difference']:.6f} Hartree")
    try:
        energy_ratio = comparison['energy_ratio']
        print(f"Energy Ratio: {energy_ratio:.6f}")
    except:
        print("Energy Ratio: Could not calculate (division by zero)")
    
    # Calculate a PES by scanning the H-H distance
    print("\nCalculating potential energy surface for anti-H2...")
    pes_data = aqc.calculate_potential_energy_surface(
        system_name="H2_antimatter",
        coordinate_index=5,  # z-coordinate of second H
        scan_range=(0.5, 2.0),
        n_points=4
    )
    
    print("\nPES Scan Results:")
    for coord, energy in zip(pes_data['coordinates'], pes_data['energies']):
        print(f"H-H Distance: {coord:.2f} Bohr, Energy: {energy:.6f} Hartree")
    
    print("\nExamining annihilation properties for mixed H2...")
    mixed_ann = mixed_result.get('annihilation_properties', {})
    for key, value in mixed_ann.items():
        print(f"  {key}: {value}")
    
    return aqc

if __name__ == "__main__":
    run_h2_example()