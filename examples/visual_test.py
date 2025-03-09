#!/usr/bin/env python3
"""
Script to save visualizations of antimatter quantum chemistry results to files.
"""

import sys
import os
import numpy as np
import matplotlib
# Force matplotlib to use a non-interactive backend
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the antimatter quantum chemistry package
from antimatter_qchem import AntimatterQuantumChemistry

def save_visualizations():
    """Generate and save visualizations of the framework to files."""
    print("=" * 60)
    print("GENERATING ANTIMATTER QUANTUM CHEMISTRY VISUALIZATIONS")
    print("=" * 60)
    
    # Create output directory if it doesn't exist
    if not os.path.exists('visualizations'):
        os.makedirs('visualizations')
        print("Created 'visualizations' directory")
    
    # Create the framework
    print("\nInitializing Antimatter Quantum Chemistry Framework...")
    aqc = AntimatterQuantumChemistry()
    
    # 1. First, create and solve a simple molecule
    print("\nCreating and solving H2 molecule...")
    
    # Normal H2
    bond_distance = 0.74  # equilibrium bond distance in Bohr
    normal_h2 = aqc.create_antimatter_system(
        name="H2_visualization",
        nuclei=[
            ("H", 1, 1.0, np.array([0.0, 0.0, 0.0]), False),
            ("H", 1, 1.0, np.array([0.0, 0.0, bond_distance]), False)
        ],
        n_electrons=2,
        n_positrons=0
    )
    
    # Anti-H2
    anti_h2 = aqc.create_antimatter_system(
        name="Anti_H2_visualization",
        nuclei=[
            ("H", 1, -1.0, np.array([0.0, 0.0, 0.0]), True),
            ("H", 1, -1.0, np.array([0.0, 0.0, bond_distance]), True)
        ],
        n_electrons=0,
        n_positrons=2
    )
    
    # Solve both systems
    print("Solving normal H2...")
    normal_result = aqc.solve_system("H2_visualization", method="classical")
    print(f"Normal H2 Energy: {normal_result['energy']:.6f} Hartree")
    
    print("Solving anti-H2...")
    anti_result = aqc.solve_system("Anti_H2_visualization", method="classical")
    print(f"Anti-H2 Energy: {anti_result['energy']:.6f} Hartree")
    
    # 2. Calculate a detailed potential energy surface
    print("\nCalculating detailed potential energy surface for H2...")
    pes_data = aqc.calculate_potential_energy_surface(
        system_name="H2_visualization",
        coordinate_index=5,  # z-coordinate of second H
        scan_range=(0.3, 3.0),
        n_points=10  # more points for smoother curve
    )
    
    # 3. Calculate the same PES for anti-H2
    print("\nCalculating potential energy surface for anti-H2...")
    anti_pes_data = aqc.calculate_potential_energy_surface(
        system_name="Anti_H2_visualization",
        coordinate_index=5,  # z-coordinate of second H
        scan_range=(0.3, 3.0),
        n_points=10
    )
    
    # 4. Compare matter and antimatter
    print("\nComparing matter and antimatter H2...")
    comparison = aqc.compare_matter_antimatter("H2_visualization")
    
    # Now create and save the visualizations manually
    print("\n" + "=" * 60)
    print("SAVING VISUALIZATIONS TO FILES")
    print("=" * 60)
    
    # 1. Visualize H2 potential energy surface
    print("\nSaving H2 potential energy surface...")
    plt.figure(figsize=(10, 6))
    plt.plot(pes_data['coordinates'], pes_data['energies'], 'b-o')
    plt.title("H2 Potential Energy Surface")
    plt.xlabel("H-H Distance (Bohr)")
    plt.ylabel("Energy (Hartree)")
    plt.grid(True)
    plt.savefig('visualizations/h2_pes.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Visualize anti-H2 potential energy surface
    print("Saving anti-H2 potential energy surface...")
    plt.figure(figsize=(10, 6))
    plt.plot(anti_pes_data['coordinates'], anti_pes_data['energies'], 'r-o')
    plt.title("Anti-H2 Potential Energy Surface")
    plt.xlabel("H-H Distance (Bohr)")
    plt.ylabel("Energy (Hartree)")
    plt.grid(True)
    plt.savefig('visualizations/anti_h2_pes.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Compare H2 and anti-H2 PES on the same plot
    print("Saving PES comparison...")
    plt.figure(figsize=(10, 6))
    plt.plot(pes_data['coordinates'], pes_data['energies'], 'b-o', label='Normal H2')
    plt.plot(anti_pes_data['coordinates'], anti_pes_data['energies'], 'r-o', label='Anti-H2')
    plt.title("Matter vs Antimatter Potential Energy Surface")
    plt.xlabel("Bond Distance (Bohr)")
    plt.ylabel("Energy (Hartree)")
    plt.legend()
    plt.grid(True)
    plt.savefig('visualizations/pes_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Matter-antimatter energy comparison
    print("Saving matter-antimatter energy comparison...")
    plt.figure(figsize=(10, 6))
    labels = [comparison['original_type'], comparison['comparison_type']]
    energies = [comparison['original_energy'], comparison['comparison_energy']]
    plt.bar(labels, energies, color=['blue', 'red'])
    plt.title("Matter vs Antimatter Energy Comparison")
    plt.ylabel("Energy (Hartree)")
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Add the energy difference as text
    plt.text(0.5, min(energies) - 0.1, 
            f"Energy Difference: {comparison['energy_difference']:.4f} Hartree",
            horizontalalignment='center')
    
    plt.savefig('visualizations/matter_antimatter_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Create an advanced visualization - both PES curves with energy marker
    print("Saving advanced visualization...")
    plt.figure(figsize=(12, 8))
    
    # Plot both PES curves
    plt.plot(pes_data['coordinates'], pes_data['energies'], 'b-', 
             label='Normal H2', linewidth=2)
    plt.plot(anti_pes_data['coordinates'], anti_pes_data['energies'], 'r-', 
             label='Anti-H2', linewidth=2)
    
    # Mark the equilibrium points
    h2_min_idx = np.argmin(pes_data['energies'])
    anti_h2_min_idx = np.argmin(anti_pes_data['energies'])
    
    h2_min_x = pes_data['coordinates'][h2_min_idx]
    h2_min_y = pes_data['energies'][h2_min_idx]
    
    anti_h2_min_x = anti_pes_data['coordinates'][anti_h2_min_idx]
    anti_h2_min_y = anti_pes_data['energies'][anti_h2_min_idx]
    
    plt.scatter(h2_min_x, h2_min_y, s=100, c='blue', marker='o', 
                edgecolors='black', zorder=5)
    plt.scatter(anti_h2_min_x, anti_h2_min_y, s=100, c='red', marker='o', 
                edgecolors='black', zorder=5)
    
    # Annotations
    plt.annotate(f'Equilibrium: {h2_min_x:.2f} Bohr\nEnergy: {h2_min_y:.4f} Hartree',
                xy=(h2_min_x, h2_min_y), xytext=(h2_min_x+0.5, h2_min_y-0.1),
                arrowprops=dict(facecolor='blue', shrink=0.05, width=1.5, headwidth=8))
    
    plt.annotate(f'Equilibrium: {anti_h2_min_x:.2f} Bohr\nEnergy: {anti_h2_min_y:.4f} Hartree',
                xy=(anti_h2_min_x, anti_h2_min_y), xytext=(anti_h2_min_x+0.5, anti_h2_min_y+0.1),
                arrowprops=dict(facecolor='red', shrink=0.05, width=1.5, headwidth=8))
    
    # Add a visualization of the energy difference
    energy_diff = h2_min_y - anti_h2_min_y
    mid_x = (h2_min_x + anti_h2_min_x) / 2
    plt.plot([mid_x, mid_x], [h2_min_y, anti_h2_min_y], 'k--', linewidth=1.5)
    plt.text(mid_x + 0.2, (h2_min_y + anti_h2_min_y) / 2, 
             f'Energy Diff:\n{energy_diff:.4f} Hartree',
             va='center')
    
    plt.title("Antimatter vs Matter: H2 Potential Energy Surfaces", fontsize=16)
    plt.xlabel("H-H Bond Distance (Bohr)", fontsize=14)
    plt.ylabel("Energy (Hartree)", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Improve appearance
    plt.tight_layout()
    plt.savefig('visualizations/advanced_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\nAll visualizations have been saved to the 'visualizations' directory:")
    for i, file in enumerate(os.listdir('visualizations')):
        print(f"{i+1}. {file}")
    
    return aqc

if __name__ == "__main__":
    save_visualizations()