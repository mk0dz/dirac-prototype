#!/usr/bin/env python3
"""
Example script for H2O (water) in both normal and antimatter forms.
"""

import sys
import os
import numpy as np

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the antimatter quantum chemistry package
from antimatter_qchem import AntimatterQuantumChemistry

def run_h2o_example():
    """Test both normal matter and antimatter H2O systems."""
    print("=" * 60)
    print("WATER MOLECULE (H2O) EXAMPLE")
    print("=" * 60)
    
    # Create the framework
    print("Initializing Antimatter Quantum Chemistry Framework...")
    aqc = AntimatterQuantumChemistry()
    
    # Geometry parameters for H2O in Bohr
    # Oxygen at origin, hydrogens positioned at typical bond length and angle
    oh_bond = 1.8  # O-H bond length in Bohr (~0.96 Å)
    hoh_angle = 104.5 * np.pi / 180  # H-O-H angle in radians
    
    # Calculate H positions based on bond length and angle
    h1_x = oh_bond * np.sin(hoh_angle / 2)
    h1_z = oh_bond * np.cos(hoh_angle / 2)
    h2_x = -h1_x
    h2_z = h1_z
    
    print("\nCreating H2O systems...")
    # Normal H2O
    normal_h2o = aqc.create_antimatter_system(
        name="H2O_normal",
        nuclei=[
            ("O", 8, 8.0, np.array([0.0, 0.0, 0.0]), False),
            ("H", 1, 1.0, np.array([h1_x, 0.0, h1_z]), False),
            ("H", 1, 1.0, np.array([h2_x, 0.0, h2_z]), False)
        ],
        n_electrons=10,  # O(8) + H(1) + H(1)
        n_positrons=0
    )
    print(f"  Created normal H2O system with O-H bond {oh_bond} Bohr and H-O-H angle {hoh_angle*180/np.pi:.1f}°")
    
    # Anti-H2O
    anti_h2o = aqc.create_antimatter_system(
        name="H2O_antimatter",
        nuclei=[
            ("O", 8, -8.0, np.array([0.0, 0.0, 0.0]), True),
            ("H", 1, -1.0, np.array([h1_x, 0.0, h1_z]), True),
            ("H", 1, -1.0, np.array([h2_x, 0.0, h2_z]), True)
        ],
        n_electrons=0,
        n_positrons=10  # Anti-O(8) + Anti-H(1) + Anti-H(1)
    )
    print(f"  Created anti-H2O system with same geometry")
    
    # Mixed H2O (O + two anti-H)
    mixed_h2o = aqc.create_antimatter_system(
        name="H2O_mixed",
        nuclei=[
            ("O", 8, 8.0, np.array([0.0, 0.0, 0.0]), False),
            ("H", 1, -1.0, np.array([h1_x, 0.0, h1_z]), True),
            ("H", 1, -1.0, np.array([h2_x, 0.0, h2_z]), True)
        ],
        n_electrons=8,  # O(8) electrons
        n_positrons=2   # Two Anti-H(1) positrons
    )
    print(f"  Created mixed H2O system (O + two anti-H)")
    
    # Solve all systems
    print("\nSolving normal H2O...")
    normal_result = aqc.solve_system("H2O_normal", method="classical")
    print(f"Normal H2O Energy: {normal_result['energy']:.6f} Hartree")
    
    print("\nSolving anti-H2O...")
    anti_result = aqc.solve_system("H2O_antimatter", method="classical")
    print(f"Anti-H2O Energy: {anti_result['energy']:.6f} Hartree")
    
    print("\nSolving mixed H2O...")
    mixed_result = aqc.solve_system("H2O_mixed", method="classical")
    print(f"Mixed H2O Energy: {mixed_result['energy']:.6f} Hartree")
    
    # Compare matter and antimatter
    print("\nComparing matter and antimatter H2O...")
    comparison = aqc.compare_matter_antimatter("H2O_normal")
    print(f"Energy Difference: {comparison['energy_difference']:.6f} Hartree")
    try:
        energy_ratio = comparison['energy_ratio']
        print(f"Energy Ratio: {energy_ratio:.6f}")
    except:
        print("Energy Ratio: Could not calculate (division by zero)")
    
    # Calculate a PES by varying one O-H bond length
    print("\nCalculating potential energy surface for normal H2O...")
    # We'll scan the z-coordinate of the first hydrogen
    pes_data = aqc.calculate_potential_energy_surface(
        system_name="H2O_normal",
        coordinate_index=5,  # z-coordinate of first H
        scan_range=(1.0, 3.0),
        n_points=5
    )
    
    print("\nPES Scan Results (varying first O-H distance):")
    for coord, energy in zip(pes_data['coordinates'], pes_data['energies']):
        # Calculate actual O-H distance (since we're just changing z)
        actual_oh = np.sqrt(h1_x**2 + coord**2)
        print(f"O-H Distance: {actual_oh:.2f} Bohr, Energy: {energy:.6f} Hartree")
    
    # Look at annihilation properties for mixed system
    print("\nExamining annihilation properties for mixed H2O...")
    mixed_ann = mixed_result.get('annihilation_properties', {})
    for key, value in mixed_ann.items():
        print(f"  {key}: {value}")
        
    return aqc

if __name__ == "__main__":
    run_h2o_example()