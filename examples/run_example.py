#!/usr/bin/env python3
"""
Example script for using the antimatter quantum chemistry prototype.

This script demonstrates creating and solving HeH+ systems in both
normal matter and antimatter forms.
"""

import sys
import os
import numpy as np

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the antimatter quantum chemistry package
from antimatter_qchem import AntimatterQuantumChemistry

def run_example():
    """Run an example antimatter quantum chemistry simulation."""
    # Create the framework
    print("Initializing Antimatter Quantum Chemistry Framework...")
    aqc = AntimatterQuantumChemistry()
    
    print("\nCreating molecular systems...")
    # Create a HeH+ system in both matter and antimatter forms
    # Normal HeH+
    normal_heh = aqc.create_antimatter_system(
        name="HeH_normal",
        nuclei=[
            ("He", 2, 2.0, np.array([0.0, 0.0, 0.0]), False),
            ("H", 1, 1.0, np.array([0.0, 0.0, 1.5]), False)
        ],
        n_electrons=2,
        n_positrons=0
    )
    print(f"  Created normal HeH+ system with {normal_heh.n_electrons} electrons")
    
    # Anti-HeH+
    anti_heh = aqc.create_antimatter_system(
        name="HeH_antimatter",
        nuclei=[
            ("He", 2, -2.0, np.array([0.0, 0.0, 0.0]), True),
            ("H", 1, -1.0, np.array([0.0, 0.0, 1.5]), True)
        ],
        n_electrons=0,
        n_positrons=2
    )
    print(f"  Created anti-HeH+ system with {anti_heh.n_positrons} positrons")
    
    # Solve both systems
    print("\nSolving normal HeH+...")
    normal_result = aqc.solve_system("HeH_normal", method="classical")
    print("Normal HeH+ Energy:", normal_result['energy'])
    
    print("\nSolving anti-HeH+...")
    anti_result = aqc.solve_system("HeH_antimatter", method="classical")
    print("Anti-HeH+ Energy:", anti_result['energy'])
    
    # Compare matter and antimatter
    print("\nComparing matter and antimatter HeH+...")
    comparison = aqc.compare_matter_antimatter("HeH_normal")
    print(f"Energy Difference: {comparison['energy_difference']:.6f} Hartree")
    print(f"Energy Ratio: {comparison['energy_ratio']:.6f}")
    
    # Calculate a PES by scanning the H-He distance
    print("\nCalculating potential energy surface...")
    pes_data = aqc.calculate_potential_energy_surface(
        system_name="HeH_antimatter",
        coordinate_index=5,  # z-coordinate of H
        scan_range=(0.5, 3.0),
        n_points=6
    )
    
    print("\nPES Scan Results:")
    for coord, energy in zip(pes_data['coordinates'], pes_data['energies']):
        print(f"H-He Distance: {coord:.2f} Bohr, Energy: {energy:.6f} Hartree")
    
    # Visualize the results
    print("\nVisualizing results (plots will be displayed if matplotlib is available)...")
    try:
        aqc.visualize_result("HeH_normal_matter_antimatter_comparison")
        aqc.visualize_result("HeH_antimatter_pes_scan_5")
    except Exception as e:
        print(f"Visualization error: {e}")
    
    print("\nExample complete! The antimatter quantum chemistry module works.")
    return aqc

if __name__ == "__main__":
    run_example()