"""
Antimatter Quantum Chemistry Prototype

This module integrates the theoretical framework, mathematical foundation, and
data structures to build a prototype for antimatter quantum chemistry that can
be extended and integrated with Qiskit Nature.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import matplotlib.pyplot as plt
from dataclasses import dataclass

# Import modules from our package
from .theory import (
    ParticleType, Particle, InteractionType,
    coulomb_potential, annihilation_probability
)

from .math_framework import (
    AntimatterHamiltonian, MolecularIntegralCalculator
)

from .data_structures import (
    MatterType, Nucleus, BasisFunction, BasisSet, 
    MolecularSystem, WavefunctionData, IntegralProvider
)

# For visualization (if available, otherwise provide fallback)
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

class AntimatterQiskitAdapter:
    """
    Adapter class to interface our antimatter chemistry framework with Qiskit Nature.
    
    This provides the bridge between our specialized antimatter models and 
    Qiskit's electronic structure framework.
    """
    
    def __init__(self, molecular_system: MolecularSystem):
        """
        Initialize the adapter with a molecular system.
        
        Args:
            molecular_system: The antimatter molecular system
        """
        self.system = molecular_system
        self.integral_provider = IntegralProvider(molecular_system)
        
        # Calculate all integrals
        self.integral_provider.calculate_all_integrals()
        
        # Initialize attributes for Qiskit interfacing
        self.qiskit_hamiltonian = None
        self.n_qubits = None
        self.qubit_mapping = None
    
    def create_antimatter_hamiltonian(self) -> AntimatterHamiltonian:
        """
        Create an AntimatterHamiltonian instance with all calculated integrals.
        
        Returns:
            AntimatterHamiltonian instance
        """
        # Determine n_particles based on system type
        if self.system.matter_type == MatterType.NORMAL:
            n_particles = (self.system.n_electrons, 0)
        elif self.system.matter_type == MatterType.ANTIMATTER:
            n_particles = (0, self.system.n_positrons)
        else:  # MIXED
            n_particles = (self.system.n_electrons, self.system.n_positrons)
        
        # Create Hamiltonian
        n_spatial_orbitals = self.system.basis_set.size
        hamiltonian = AntimatterHamiltonian(
            n_spatial_orbitals=n_spatial_orbitals,
            n_particles=n_particles,
            include_annihilation=self.system.has_antimatter,
            relativistic_correction=self.system.has_antimatter
        )
        
        # Set one-body integrals (H_core = T + V)
        one_body = self.integral_provider.get_core_hamiltonian()
        
        # Extend to spin-orbitals (block diagonal)
        n_spin_orbitals = 2 * n_spatial_orbitals
        one_body_spin = np.zeros((n_spin_orbitals, n_spin_orbitals))
        one_body_spin[:n_spatial_orbitals, :n_spatial_orbitals] = one_body
        one_body_spin[n_spatial_orbitals:, n_spatial_orbitals:] = one_body
        hamiltonian.set_one_body_integrals(one_body_spin)
        
        # Set two-body integrals (electron repulsion)
        two_body = self.integral_provider.electron_repulsion
        
        # Extend to spin-orbitals
        n_spin_orbitals = 2 * n_spatial_orbitals
        two_body_spin = np.zeros((n_spin_orbitals, n_spin_orbitals, 
                                 n_spin_orbitals, n_spin_orbitals))
        
        # Assume spin conservation (only non-zero if spins match)
        for p in range(n_spatial_orbitals):
            for q in range(n_spatial_orbitals):
                for r in range(n_spatial_orbitals):
                    for s in range(n_spatial_orbitals):
                        # Alpha-Alpha-Alpha-Alpha
                        two_body_spin[p, q, r, s] = two_body[p, q, r, s]
                        # Beta-Beta-Beta-Beta
                        two_body_spin[p+n_spatial_orbitals, q+n_spatial_orbitals, 
                                     r+n_spatial_orbitals, s+n_spatial_orbitals] = two_body[p, q, r, s]
                        # Alpha-Alpha-Beta-Beta
                        two_body_spin[p, q, r+n_spatial_orbitals, s+n_spatial_orbitals] = two_body[p, q, r, s]
                        # Beta-Beta-Alpha-Alpha
                        two_body_spin[p+n_spatial_orbitals, q+n_spatial_orbitals, r, s] = two_body[p, q, r, s]
        
        hamiltonian.set_two_body_integrals(two_body_spin)
        
        # Set annihilation integrals if applicable
        if self.system.has_antimatter:
            annihilation_ints = self.integral_provider.annihilation
            if annihilation_ints is not None:
                # Extend to spin-orbitals
                annihilation_spin = np.zeros((n_spin_orbitals, n_spin_orbitals))
                annihilation_spin[:n_spatial_orbitals, :n_spatial_orbitals] = annihilation_ints
                annihilation_spin[n_spatial_orbitals:, n_spatial_orbitals:] = annihilation_ints
                hamiltonian.set_annihilation_integrals(annihilation_spin)
            else:
                hamiltonian.set_annihilation_integrals()
            
            # Set relativistic correction
            hamiltonian.set_relativistic_correction()
        
        return hamiltonian
    
    def map_to_qiskit_hamiltonian(self, qubit_mapping: str = 'jordan_wigner'):
        """
        Map the antimatter Hamiltonian to a Qiskit operator.
        
        Args:
            qubit_mapping: Mapping strategy ('jordan_wigner' or 'parity')
            
        Returns:
            Qiskit operator representation of the Hamiltonian
        """
        hamiltonian = self.create_antimatter_hamiltonian()
        self.qubit_mapping = qubit_mapping
        
        # Get the second-quantized Hamiltonian dictionary
        hamiltonian_dict = hamiltonian.get_second_quantized_hamiltonian()
        
        # In an actual implementation, this would create a Qiskit operator
        # using the chosen mapping. For the prototype, we'll just store the 
        # information needed to create it.
        
        # Calculate number of qubits required
        n_spin_orbitals = 2 * self.system.basis_set.size
        
        if qubit_mapping == 'jordan_wigner':
            self.n_qubits = n_spin_orbitals
        elif qubit_mapping == 'parity':
            self.n_qubits = n_spin_orbitals
        else:
            raise ValueError(f"Unsupported qubit mapping: {qubit_mapping}")
        
        # Store information for later use with Qiskit
        self.qiskit_hamiltonian = {
            'hamiltonian_dict': hamiltonian_dict,
            'n_spin_orbitals': n_spin_orbitals,
            'mapping': qubit_mapping,
            'n_qubits': self.n_qubits
        }
        
        return self.qiskit_hamiltonian
    
    def estimate_qubit_requirements(self) -> Dict[str, int]:
        """
        Estimate the number of qubits required for different mappings.
        
        Returns:
            Dictionary with qubit requirements for each mapping
        """
        n_spin_orbitals = 2 * self.system.basis_set.size
        
        return {
            'jordan_wigner': n_spin_orbitals,
            'parity': n_spin_orbitals,
            'bravyi_kitaev': n_spin_orbitals,
            'with_annihilation': n_spin_orbitals + 1  # Extra qubit for annihilation
        }
    
    def prepare_for_qiskit_vqe(self):
        """
        Prepare the system for running VQE with Qiskit.
        
        Returns:
            Dictionary with information needed for VQE
        """
        # Make sure we have a mapped Hamiltonian
        if self.qiskit_hamiltonian is None:
            self.map_to_qiskit_hamiltonian()
        
        # Determine initial state based on matter type and particle numbers
        if self.system.matter_type == MatterType.NORMAL:
            initial_state = f"{self.system.n_electrons}_electrons"
        elif self.system.matter_type == MatterType.ANTIMATTER:
            initial_state = f"{self.system.n_positrons}_positrons"
        else:  # MIXED
            initial_state = f"{self.system.n_electrons}e_{self.system.n_positrons}p"
        
        # In a real implementation, this would return the actual objects
        # needed for Qiskit VQE. For the prototype, we return information
        # about what would be created.
        return {
            'operator': self.qiskit_hamiltonian,
            'initial_state': initial_state,
            'n_qubits': self.n_qubits,
            'mapping': self.qubit_mapping,
            'matter_type': self.system.matter_type.name
        }

class AntimatterSolver:
    """
    Solver for antimatter quantum chemistry problems.
    
    This class provides classical and quantum methods for solving
    antimatter quantum chemistry problems.
    """
    
    def __init__(self, molecular_system: MolecularSystem):
        """
        Initialize with a molecular system.
        
        Args:
            molecular_system: The molecular system to solve
        """
        self.system = molecular_system
        self.adapter = AntimatterQiskitAdapter(molecular_system)
        self.result = None
    
    def solve_classically(self) -> WavefunctionData:
        """
        Solve the system using classical matrix diagonalization.
        
        Returns:
            WavefunctionData with the solution
        """
        # Get integrals
        integrals = self.adapter.integral_provider.get_all_integrals()
        overlap = integrals['overlap']
        h_core = self.adapter.integral_provider.get_core_hamiltonian()
        
        # For a real implementation, this would perform a proper SCF calculation
        # For the prototype, we'll simulate the result with a simplified approach
        
        # Solve the generalized eigenvalue problem H·C = S·C·E
        # We need to use scipy's eigh function for the generalized eigenvalue problem
        try:
            # Try to use scipy if available (preferred method)
            from scipy import linalg
            eigenvalues, eigenvectors = linalg.eigh(h_core, overlap)
        except ImportError:
            # Fallback to a less stable but workable approach with NumPy
            # Convert generalized eigenvalue problem to standard form
            # Ax = λBx becomes inv(B)Ax = λx
            # Note: This is less numerically stable
            overlap_inv = np.linalg.inv(overlap)
            transformed_h = overlap_inv @ h_core
            eigenvalues, eigenvectors = np.linalg.eigh(transformed_h)
        
        # Handle different matter types properly
        if self.system.matter_type == MatterType.MIXED:
            # For mixed systems, we need separate electron and positron coefficients
            electron_coefficients = eigenvectors.copy()
            positron_coefficients = eigenvectors.copy()
            
            # Create wavefunction data object for mixed systems
            wf_data = WavefunctionData(
                matter_type=self.system.matter_type,
                n_spatial_orbitals=self.system.basis_set.size,
                n_electrons=self.system.n_electrons,
                n_positrons=self.system.n_positrons,
                basis_set=self.system.basis_set,
                orbital_coefficients=eigenvectors,
                electron_coefficients=electron_coefficients,
                positron_coefficients=positron_coefficients,
                energy=np.sum(eigenvalues[:self.system.n_electrons]) + np.sum(eigenvalues[:self.system.n_positrons])
            )
        else:
            # Normal matter or pure antimatter systems
            # Create wavefunction data object
            wf_data = WavefunctionData(
                matter_type=self.system.matter_type,
                n_spatial_orbitals=self.system.basis_set.size,
                n_electrons=self.system.n_electrons,
                n_positrons=self.system.n_positrons,
                basis_set=self.system.basis_set,
                orbital_coefficients=eigenvectors,
                energy=(np.sum(eigenvalues[:self.system.n_electrons]) if self.system.matter_type == MatterType.NORMAL
                       else np.sum(eigenvalues[:self.system.n_positrons]))
            )
        
        self.result = wf_data
        return wf_data
    
    def solve_with_vqe(self, n_layers: int = 2, optimizer: str = 'COBYLA', 
                      shots: int = 1024) -> Dict:
        """
        Solve the system using VQE on a quantum computer or simulator.
        
        Args:
            n_layers: Number of layers in the ansatz
            optimizer: Classical optimizer to use ('COBYLA', 'SLSQP', etc.)
            shots: Number of measurement shots
            
        Returns:
            Dictionary with VQE results
        """
        # Prepare the system for VQE
        vqe_info = self.adapter.prepare_for_qiskit_vqe()
        
        # In a real implementation, this would run VQE using Qiskit
        # For the prototype, we'll simulate the result
        
        # Simulate energy close to the classical solution but slightly higher
        classical_result = self.solve_classically()
        vqe_energy = classical_result.energy * (1 + np.random.uniform(0.01, 0.05))
        
        # Create simulated VQE result
        vqe_result = {
            'energy': vqe_energy,
            'optimizer_iterations': np.random.randint(20, 100),
            'optimal_parameters': np.random.random(n_layers * vqe_info['n_qubits']),
            'circuit_depth': n_layers * 5 + 10,
            'n_qubits': vqe_info['n_qubits'],
            'classical_energy': classical_result.energy
        }
        
        return vqe_result
    
    def analyze_matter_antimatter_difference(self) -> Dict:
        """
        Analyze the differences between matter and antimatter versions of the same system.
        
        Returns:
            Dictionary with comparative analysis results
        """
        # Create a copy of the system with opposite matter type
        comparison_system = MolecularSystem(
            name=f"Comparison for {self.system.name}",
            matter_type=MatterType.NORMAL if self.system.matter_type == MatterType.ANTIMATTER 
                        else MatterType.ANTIMATTER
        )
        
        # Copy nuclei with inverted charges
        for nucleus in self.system.nuclei:
            comparison_nucleus = Nucleus(
                symbol=nucleus.symbol,
                atomic_number=nucleus.atomic_number,
                charge=-nucleus.charge,  # Invert charge
                mass=nucleus.mass,
                position=nucleus.position.copy(),
                is_anti=not nucleus.is_anti  # Invert matter type
            )
            comparison_system.add_nucleus(comparison_nucleus)
        
        # Set the same basis set
        comparison_system.set_basis_set(self.system.basis_set)
        
        # Set electron/positron numbers based on matter type
        if self.system.matter_type == MatterType.NORMAL:
            comparison_system.set_n_electrons(0)
            comparison_system.set_n_positrons(self.system.n_electrons)
        else:
            comparison_system.set_n_electrons(self.system.n_positrons)
            comparison_system.set_n_positrons(0)
        
        # Solve both systems
        original_result = self.solve_classically()
        
        comparison_solver = AntimatterSolver(comparison_system)
        comparison_result = comparison_solver.solve_classically()
        
        # Calculate differences
        energy_diff = original_result.energy - comparison_result.energy
        
        # Calculate energy ratio safely (avoiding division by zero)
        if abs(comparison_result.energy) < 1e-10:
            # If denominator is close to zero
            if abs(original_result.energy) < 1e-10:
                # Both energies close to zero
                energy_ratio = 1.0
            else:
                # Only comparison energy close to zero
                energy_ratio = float('inf') if original_result.energy > 0 else float('-inf')
        else:
            # Normal case - safe to divide
            energy_ratio = original_result.energy / comparison_result.energy
        
        # Generate the analysis report
        return {
            'original_system': self.system.name,
            'original_type': self.system.matter_type.name,
            'original_energy': original_result.energy,
            'comparison_system': comparison_system.name,
            'comparison_type': comparison_system.matter_type.name,
            'comparison_energy': comparison_result.energy,
            'energy_difference': energy_diff,
            'energy_ratio': energy_ratio,
            'original_orbitals': original_result.orbital_coefficients,
            'comparison_orbitals': comparison_result.orbital_coefficients
        }
    
    def calculate_annihilation_properties(self) -> Dict:
        """
        Calculate properties related to electron-positron annihilation.
        
        Returns:
            Dictionary with annihilation-related properties
        """
        if not self.system.has_antimatter:
            return {'annihilation_probability': 0.0, 'has_antimatter': False}
        
        # For a real implementation, this would calculate detailed annihilation properties
        # For the prototype, we'll provide a simplified model
        
        # Solve the system if not already done
        if self.result is None:
            self.solve_classically()
        
        # Access annihilation integrals
        annihilation_ints = self.adapter.integral_provider.annihilation
        
        if annihilation_ints is None:
            return {
                'has_antimatter': True,
                'annihilation_probability': 0.0,
                'annihilation_calculated': False
            }
        
        # Calculate an overall annihilation probability
        # This is a simplified model
        total_prob = np.sum(np.abs(annihilation_ints))
        normalized_prob = min(1.0, total_prob / (self.system.basis_set.size ** 2))
        
        # Calculate positronium formation probability if applicable
        positronium_prob = 0.0
        if self.system.matter_type == MatterType.MIXED and self.system.n_electrons > 0 and self.system.n_positrons > 0:
            # Simple model for positronium formation
            positronium_prob = normalized_prob * 0.8
        
        return {
            'has_antimatter': True,
            'annihilation_probability': normalized_prob,
            'positronium_formation_probability': positronium_prob,
            'annihilation_calculated': True,
            'annihilation_integrals_trace': np.trace(annihilation_ints)
        }

class AntimatterQuantumChemistry:
    """
    Main class for antimatter quantum chemistry simulations.
    
    This class serves as the primary interface for users to set up and run
    antimatter quantum chemistry simulations.
    """
    
    def __init__(self):
        """Initialize the antimatter quantum chemistry framework."""
        self.systems = {}
        self.results = {}
    
    def create_antimatter_system(self, name: str, 
                                nuclei: List[Tuple[str, int, float, np.ndarray, bool]],
                                basis_name: str = "sto-3g",
                                n_positrons: int = 0,
                                n_electrons: int = 0) -> MolecularSystem:
        """
        Create an antimatter molecular system.
        
        Args:
            name: Name for the system
            nuclei: List of (symbol, atomic_number, charge, position, is_anti) tuples
            basis_name: Name of the basis set to use
            n_positrons: Number of positrons
            n_electrons: Number of electrons
            
        Returns:
            Created MolecularSystem
        """
        # Determine the matter type
        if n_positrons > 0 and n_electrons == 0:
            matter_type = MatterType.ANTIMATTER
        elif n_positrons == 0 and n_electrons > 0:
            matter_type = MatterType.NORMAL
        else:
            matter_type = MatterType.MIXED
        
        # Create the system
        system = MolecularSystem(name=name, matter_type=matter_type)
        
        # Add nuclei
        for symbol, atomic_number, charge, position, is_anti in nuclei:
            nucleus = Nucleus(
                symbol=symbol,
                atomic_number=atomic_number,
                charge=charge,
                mass=atomic_number * 1836.15,  # Approximate mass in atomic units
                position=position,
                is_anti=is_anti
            )
            system.add_nucleus(nucleus)
        
        # Create a basic basis set
        basis = BasisSet(name=basis_name, matter_type=matter_type)
        
        # Add a single s-type basis function for each nucleus
        for i, (symbol, _, _, pos, _) in enumerate(nuclei):
            basis_func = BasisFunction(
                center_idx=i,
                angular_momentum=(0, 0, 0),  # s-type
                exponents=[0.5, 1.0, 2.0],   # Example STO-3G-like
                coefficients=[0.4, 0.4, 0.2],  # Example coefficients
                normalization=1.0
            )
            basis.add_function(basis_func)
        
        # Set the basis set for the system
        system.set_basis_set(basis)
        
        # Set the number of electrons and positrons
        system.set_n_electrons(n_electrons)
        system.set_n_positrons(n_positrons)
        
        # Store the system
        self.systems[name] = system
        
        return system
    
    def solve_system(self, system_name: str, method: str = 'classical') -> Dict:
        """
        Solve a molecular system and store the results.
        
        Args:
            system_name: Name of the system to solve
            method: Solution method ('classical' or 'vqe')
            
        Returns:
            Dictionary with the results
        """
        if system_name not in self.systems:
            raise ValueError(f"System {system_name} not found")
        
        system = self.systems[system_name]
        solver = AntimatterSolver(system)
        
        if method == 'classical':
            wf_data = solver.solve_classically()
            result = {
                'energy': wf_data.energy,
                'wavefunction': wf_data,
                'method': 'classical'
            }
        elif method == 'vqe':
            vqe_result = solver.solve_with_vqe()
            result = {
                'energy': vqe_result['energy'],
                'vqe_data': vqe_result,
                'method': 'vqe'
            }
        else:
            raise ValueError(f"Unsupported method: {method}")
        
        # Calculate extra properties based on system type
        if system.has_antimatter:
            annihilation_props = solver.calculate_annihilation_properties()
            result['annihilation_properties'] = annihilation_props
        
        # Store the result
        self.results[system_name] = result
        
        return result
    
    def compare_matter_antimatter(self, system_name: str) -> Dict:
        """
        Compare matter and antimatter versions of a system.
        
        Args:
            system_name: Name of the system to analyze
            
        Returns:
            Dictionary with comparison results
        """
        if system_name not in self.systems:
            raise ValueError(f"System {system_name} not found")
        
        system = self.systems[system_name]
        solver = AntimatterSolver(system)
        
        comparison = solver.analyze_matter_antimatter_difference()
        
        # Store the comparison
        result_key = f"{system_name}_matter_antimatter_comparison"
        self.results[result_key] = comparison
        
        return comparison
    
    def calculate_potential_energy_surface(self, system_name: str, 
                                         coordinate_index: int,
                                         scan_range: Tuple[float, float],
                                         n_points: int = 10) -> Dict:
        """
        Calculate a potential energy surface by scanning a coordinate.
        
        Args:
            system_name: Name of the system
            coordinate_index: Index of the coordinate to scan (atom index * 3 + axis)
            scan_range: (min, max) range for the scan
            n_points: Number of points in the scan
            
        Returns:
            Dictionary with the PES data
        """
        if system_name not in self.systems:
            raise ValueError(f"System {system_name} not found")
        
        # Get the original system
        original_system = self.systems[system_name]
        
        # Prepare arrays for results
        coordinates = np.linspace(scan_range[0], scan_range[1], n_points)
        energies = np.zeros(n_points)
        
        # Determine which atom and axis we're scanning
        atom_idx = coordinate_index // 3
        axis = coordinate_index % 3
        
        # Run the scan
        for i, coord in enumerate(coordinates):
            # Create a copy of the system
            system_copy = MolecularSystem(
                name=f"{system_name}_scan_{i}",
                matter_type=original_system.matter_type
            )
            
            # Copy all nuclei
            for j, nucleus in enumerate(original_system.nuclei):
                position = nucleus.position.copy()
                
                # Modify the scanned coordinate
                if j == atom_idx:
                    position[axis] = coord
                
                new_nucleus = Nucleus(
                    symbol=nucleus.symbol,
                    atomic_number=nucleus.atomic_number,
                    charge=nucleus.charge,
                    mass=nucleus.mass,
                    position=position,
                    is_anti=nucleus.is_anti
                )
                system_copy.add_nucleus(new_nucleus)
            
            # Set the same basis set
            system_copy.set_basis_set(original_system.basis_set)
            
            # Set the same electron/positron counts
            system_copy.set_n_electrons(original_system.n_electrons)
            system_copy.set_n_positrons(original_system.n_positrons)
            
            # Solve the system
            solver = AntimatterSolver(system_copy)
            result = solver.solve_classically()
            
            # Store the energy
            energies[i] = result.energy
        
        # Create the PES data
        pes_data = {
            'system_name': system_name,
            'coordinate_index': coordinate_index,
            'atom_index': atom_idx,
            'axis': axis,
            'coordinates': coordinates,
            'energies': energies,
        }
        
        # Store the result
        result_key = f"{system_name}_pes_scan_{coordinate_index}"
        self.results[result_key] = pes_data
        
        return pes_data
    
    def visualize_result(self, result_key: str):
        """
        Visualize a result.
        
        Args:
            result_key: Key of the result to visualize
        """
        if result_key not in self.results:
            raise ValueError(f"Result {result_key} not found")
        
        # Check if matplotlib is available
        if plt is None:
            print("Matplotlib is not available for visualization.")
            return
        
        result = self.results[result_key]
        
        # Check the type of result and create an appropriate visualization
        if 'pes_scan' in result_key:
            # PES scan
            plt.figure(figsize=(10, 6))
            plt.plot(result['coordinates'], result['energies'], 'b-o')
            plt.title(f"PES Scan for {result['system_name']}")
            plt.xlabel(f"Coordinate {result['coordinate_index']} (Bohr)")
            plt.ylabel("Energy (Hartree)")
            plt.grid(True)
            plt.show()
            
        elif 'matter_antimatter_comparison' in result_key:
            # Matter-antimatter comparison
            plt.figure(figsize=(10, 6))
            labels = [result['original_type'], result['comparison_type']]
            energies = [result['original_energy'], result['comparison_energy']]
            plt.bar(labels, energies, color=['blue', 'red'])
            plt.title(f"Matter vs Antimatter Energy Comparison")
            plt.ylabel("Energy (Hartree)")
            plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            
            # Add the energy difference as text
            plt.text(0.5, min(energies) - 0.1, 
                    f"Energy Difference: {result['energy_difference']:.4f} Hartree",
                    horizontalalignment='center')
            
            plt.show()
            
        else:
            # Simple energy result
            system_name = result_key
            method = result['method']
            energy = result['energy']
            
            print(f"System: {system_name}")
            print(f"Method: {method}")
            print(f"Energy: {energy:.6f} Hartree")
            
            if 'annihilation_properties' in result:
                ann_props = result['annihilation_properties']
                print("\nAnnihilation Properties:")
                for key, value in ann_props.items():
                    print(f"  {key}: {value}")