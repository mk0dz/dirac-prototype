"""
Data structures for antimatter quantum chemistry simulations

This module defines the data structures needed to represent antimatter
molecular systems, basis sets, wavefunctions, and operators.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Union, Callable
from dataclasses import dataclass
from enum import Enum

class MatterType(Enum):
    """Enumeration for different types of matter."""
    NORMAL = 1
    ANTIMATTER = 2
    MIXED = 3  # Systems with both matter and antimatter components

@dataclass
class Nucleus:
    """Data structure representing a nucleus or anti-nucleus."""
    symbol: str               # Chemical symbol
    atomic_number: int        # Z number 
    charge: float             # Effective charge (can be negative for anti-nuclei)
    mass: float               # Nuclear mass
    position: np.ndarray      # Position vector (x, y, z) in Bohr
    is_anti: bool = False     # Whether this is an anti-nucleus
    
    def __post_init__(self):
        """Validate and adjust properties after initialization."""
        if self.is_anti:
            self.charge = -self.charge  # Invert charge for anti-nuclei
            self.symbol = f"Anti-{self.symbol}"  # Prefix symbol for clarity

@dataclass
class BasisFunction:
    """Data structure representing a basis function."""
    center_idx: int                   # Index of the nucleus this basis is centered on
    angular_momentum: Tuple[int, int, int]  # Angular momentum (l, m, n)
    exponents: List[float]            # Gaussian exponents
    coefficients: List[float]         # Contraction coefficients
    normalization: float = 1.0        # Normalization constant
    
    def value(self, position: np.ndarray, nuclei_positions: List[np.ndarray]) -> float:
        """
        Evaluate the basis function at a given position.
        
        Args:
            position: The position to evaluate at
            nuclei_positions: Positions of all nuclei
            
        Returns:
            Value of the basis function
        """
        # Get center position
        center = nuclei_positions[self.center_idx]
        
        # Calculate relative position
        rel_pos = position - center
        
        # Calculate r^2
        r_squared = np.sum(rel_pos**2)
        
        # Calculate angular part (x^l * y^m * z^n)
        l, m, n = self.angular_momentum
        angular = rel_pos[0]**l * rel_pos[1]**m * rel_pos[2]**n
        
        # Calculate radial part (linear combination of Gaussians)
        radial = 0.0
        for exponent, coefficient in zip(self.exponents, self.coefficients):
            radial += coefficient * np.exp(-exponent * r_squared)
        
        return self.normalization * angular * radial

class BasisSet:
    """Container for a complete basis set for a molecular system."""
    
    def __init__(self, name: str, matter_type: MatterType = MatterType.NORMAL):
        """
        Initialize a basis set.
        
        Args:
            name: Name of the basis set
            matter_type: Type of matter this basis set is designed for
        """
        self.name = name
        self.matter_type = matter_type
        self.functions: List[BasisFunction] = []
        self.nuclei_positions: List[np.ndarray] = []
        
    def add_function(self, basis_function: BasisFunction) -> None:
        """Add a basis function to the set."""
        self.functions.append(basis_function)
    
    def set_nuclei_positions(self, positions: List[np.ndarray]) -> None:
        """Set the positions of all nuclei."""
        self.nuclei_positions = positions
    
    def get_function_value(self, function_idx: int, position: np.ndarray) -> float:
        """
        Get the value of a basis function at a specific position.
        
        Args:
            function_idx: Index of the function
            position: Position to evaluate at
            
        Returns:
            Value of the basis function
        """
        return self.functions[function_idx].value(position, self.nuclei_positions)
    
    def get_all_function_values(self, position: np.ndarray) -> np.ndarray:
        """
        Get values of all basis functions at a specific position.
        
        Args:
            position: Position to evaluate at
            
        Returns:
            Array of basis function values
        """
        return np.array([f.value(position, self.nuclei_positions) 
                         for f in self.functions])
    
    @property
    def size(self) -> int:
        """Get the number of basis functions."""
        return len(self.functions)

class MolecularSystem:
    """Representation of a molecular system that can be normal matter or antimatter."""
    
    def __init__(self, 
                 name: str, 
                 matter_type: MatterType = MatterType.NORMAL):
        """
        Initialize a molecular system.
        
        Args:
            name: Name of the system
            matter_type: Type of matter in this system
        """
        self.name = name
        self.matter_type = matter_type
        self.nuclei: List[Nucleus] = []
        self.basis_set: Optional[BasisSet] = None
        self.n_electrons: int = 0
        self.n_positrons: int = 0
        
        # Properties will be set later
        self.hamiltonian = None
        self.integrals = {}
        
    def add_nucleus(self, nucleus: Nucleus) -> None:
        """Add a nucleus to the system."""
        # Ensure consistency between nucleus type and system type
        if nucleus.is_anti and self.matter_type == MatterType.NORMAL:
            self.matter_type = MatterType.MIXED
        elif not nucleus.is_anti and self.matter_type == MatterType.ANTIMATTER:
            self.matter_type = MatterType.MIXED
            
        self.nuclei.append(nucleus)
    
    def set_basis_set(self, basis_set: BasisSet) -> None:
        """Set the basis set for this system."""
        self.basis_set = basis_set
        # Update the nuclei positions in the basis set
        self.basis_set.set_nuclei_positions([n.position for n in self.nuclei])
    
    def set_n_electrons(self, n: int) -> None:
        """Set the number of electrons."""
        self.n_electrons = n
    
    def set_n_positrons(self, n: int) -> None:
        """Set the number of positrons."""
        self.n_positrons = n
        
    @property
    def total_nuclear_charge(self) -> float:
        """Calculate the total nuclear charge."""
        return sum(nucleus.charge for nucleus in self.nuclei)
    
    @property
    def nuclear_positions(self) -> List[np.ndarray]:
        """Get positions of all nuclei."""
        return [nucleus.position for nucleus in self.nuclei]
    
    @property
    def nuclear_charges(self) -> List[float]:
        """Get charges of all nuclei."""
        return [nucleus.charge for nucleus in self.nuclei]
    
    @property
    def n_nuclei(self) -> int:
        """Get the number of nuclei."""
        return len(self.nuclei)
    
    @property
    def has_antimatter(self) -> bool:
        """Determine if the system contains any antimatter components."""
        return (self.matter_type in [MatterType.ANTIMATTER, MatterType.MIXED] or
                self.n_positrons > 0 or
                any(nucleus.is_anti for nucleus in self.nuclei))

@dataclass
class WavefunctionData:
    """
    Data structure to store wavefunction information.
    
    For antimatter systems, we need to track both electron and positron states.
    """
    # System description
    matter_type: MatterType
    n_spatial_orbitals: int
    n_electrons: int
    n_positrons: int
    
    # Basis information
    basis_set: BasisSet
    
    # Wavefunction representation
    orbital_coefficients: np.ndarray     # Shape: (n_basis, n_orbitals)
    
    # For mixed systems: separate coefficients for electrons and positrons
    electron_coefficients: Optional[np.ndarray] = None  # Shape: (n_basis, n_electron_orbitals)
    positron_coefficients: Optional[np.ndarray] = None  # Shape: (n_basis, n_positron_orbitals)
    
    # Energy and other properties
    energy: float = 0.0
    
    def __post_init__(self):
        """Initialize optional attributes if needed."""
        if self.matter_type == MatterType.MIXED:
            # For mixed systems, make sure we have separate coefficients
            if self.electron_coefficients is None or self.positron_coefficients is None:
                raise ValueError("Mixed systems require separate electron and positron coefficients")
    
    def get_density_matrix(self) -> np.ndarray:
        """
        Calculate the one-particle density matrix.
        
        For normal matter or pure antimatter, this is straightforward.
        For mixed systems, this returns the total density.
        
        Returns:
            Density matrix P
        """
        if self.matter_type == MatterType.MIXED:
            # For mixed systems, combine electron and positron densities
            n_basis = self.basis_set.size
            P = np.zeros((n_basis, n_basis))
            
            # Add electron contribution
            occupied_e = min(self.n_electrons, self.electron_coefficients.shape[1])
            for i in range(occupied_e):
                c = self.electron_coefficients[:, i]
                P += np.outer(c, c)
            
            # Add positron contribution
            occupied_p = min(self.n_positrons, self.positron_coefficients.shape[1])
            for i in range(occupied_p):
                c = self.positron_coefficients[:, i]
                P += np.outer(c, c)
            
            return P
        else:
            # For pure matter or antimatter systems
            P = np.zeros((self.n_spatial_orbitals, self.n_spatial_orbitals))
            occupied = self.n_electrons if self.matter_type == MatterType.NORMAL else self.n_positrons
            occupied = min(occupied, self.orbital_coefficients.shape[1])
            
            for i in range(occupied):
                c = self.orbital_coefficients[:, i]
                P += np.outer(c, c)
            
            return P
    
    def get_orbital_value(self, orbital_idx: int, position: np.ndarray) -> float:
        """
        Calculate the value of a molecular orbital at a specific position.
        
        Args:
            orbital_idx: Index of the orbital
            position: Position to evaluate at
            
        Returns:
            Value of the orbital
        """
        # Get values of all basis functions at this position
        basis_values = self.basis_set.get_all_function_values(position)
        
        # Get coefficients for this orbital
        coefficients = self.orbital_coefficients[:, orbital_idx]
        
        # Calculate orbital value as linear combination of basis functions
        return np.sum(coefficients * basis_values)
    
    def get_electron_density(self, position: np.ndarray) -> float:
        """
        Calculate the electron density at a specific position.
        
        Args:
            position: Position to evaluate at
            
        Returns:
            Electron density
        """
        density = 0.0
        
        if self.matter_type == MatterType.MIXED:
            # For mixed systems, use electron coefficients
            basis_values = self.basis_set.get_all_function_values(position)
            occupied = min(self.n_electrons, self.electron_coefficients.shape[1])
            
            for i in range(occupied):
                orbital_value = np.sum(self.electron_coefficients[:, i] * basis_values)
                density += orbital_value**2
        elif self.matter_type == MatterType.NORMAL:
            # For normal matter, use regular coefficients
            basis_values = self.basis_set.get_all_function_values(position)
            occupied = min(self.n_electrons, self.orbital_coefficients.shape[1])
            
            for i in range(occupied):
                orbital_value = np.sum(self.orbital_coefficients[:, i] * basis_values)
                density += orbital_value**2
        
        return density
    
    def get_positron_density(self, position: np.ndarray) -> float:
        """
        Calculate the positron density at a specific position.
        
        Args:
            position: Position to evaluate at
            
        Returns:
            Positron density
        """
        if self.matter_type == MatterType.NORMAL:
            return 0.0  # No positrons in normal matter
        
        density = 0.0
        
        if self.matter_type == MatterType.MIXED:
            # For mixed systems, use positron coefficients
            basis_values = self.basis_set.get_all_function_values(position)
            occupied = min(self.n_positrons, self.positron_coefficients.shape[1])
            
            for i in range(occupied):
                orbital_value = np.sum(self.positron_coefficients[:, i] * basis_values)
                density += orbital_value**2
        elif self.matter_type == MatterType.ANTIMATTER:
            # For antimatter, use regular coefficients
            basis_values = self.basis_set.get_all_function_values(position)
            occupied = min(self.n_positrons, self.orbital_coefficients.shape[1])
            
            for i in range(occupied):
                orbital_value = np.sum(self.orbital_coefficients[:, i] * basis_values)
                density += orbital_value**2
        
        return density

class IntegralProvider:
    """Interface for calculating and storing molecular integrals."""
    
    def __init__(self, molecular_system: MolecularSystem):
        """
        Initialize with a molecular system.
        
        Args:
            molecular_system: The molecular system to calculate integrals for
        """
        self.system = molecular_system
        
        # Storage for different types of integrals
        self.overlap = None              # Overlap integrals (S)
        self.kinetic = None              # Kinetic energy integrals (T)
        self.nuclear = None              # Nuclear attraction integrals (V)
        self.electron_repulsion = None   # Electron repulsion integrals (ERI)
        self.annihilation = None         # Annihilation integrals (A)
        
    def calculate_all_integrals(self) -> None:
        """Calculate all required integrals for the system."""
        self.calculate_overlap_integrals()
        self.calculate_kinetic_integrals()
        self.calculate_nuclear_integrals()
        self.calculate_repulsion_integrals()
        
        # Only calculate annihilation integrals for systems with antimatter
        if self.system.has_antimatter:
            self.calculate_annihilation_integrals()
    
    def calculate_overlap_integrals(self) -> np.ndarray:
        """
        Calculate overlap integrals.
        
        Returns:
            Overlap matrix S
        """
        # Placeholder - in a real implementation, this would compute accurate integrals
        n_basis = self.system.basis_set.size
        self.overlap = np.eye(n_basis)  # Simplified: just use identity matrix
        return self.overlap
    
    def calculate_kinetic_integrals(self) -> np.ndarray:
        """
        Calculate kinetic energy integrals.
        
        Returns:
            Kinetic energy matrix T
        """
        # Placeholder - in a real implementation, this would compute accurate integrals
        n_basis = self.system.basis_set.size
        self.kinetic = np.diag(np.arange(n_basis) * 0.1)  # Simplified diagonal matrix
        return self.kinetic
    
    def calculate_nuclear_integrals(self) -> np.ndarray:
        """
        Calculate nuclear attraction integrals.
        
        For antimatter, these have opposite sign compared to normal matter.
        
        Returns:
            Nuclear attraction matrix V
        """
        # Placeholder - in a real implementation, this would compute accurate integrals
        n_basis = self.system.basis_set.size
        
        # Create a simple model where nuclear attraction depends on basis function index
        # and nuclear charges
        V = np.zeros((n_basis, n_basis))
        total_nuclear_charge = self.system.total_nuclear_charge
        
        for i in range(n_basis):
            for j in range(n_basis):
                # Simple model: attraction proportional to basis function overlap
                # and total nuclear charge
                V[i, j] = -total_nuclear_charge * np.exp(-(i-j)**2 / 10.0) * 0.1
        
        # For antimatter, invert the sign
        if self.system.matter_type == MatterType.ANTIMATTER:
            V = -V
        
        self.nuclear = V
        return V
    
    def calculate_repulsion_integrals(self) -> np.ndarray:
        """
        Calculate electron repulsion integrals.
        
        Returns:
            Electron repulsion integral tensor ERI
        """
        # Placeholder - in a real implementation, this would compute accurate integrals
        n_basis = self.system.basis_set.size
        
        # Create a simplified ERI tensor
        ERI = np.zeros((n_basis, n_basis, n_basis, n_basis))
        
        for i in range(n_basis):
            for j in range(n_basis):
                for k in range(n_basis):
                    for l in range(n_basis):
                        # Simple model: repulsion inversely proportional to orbital separation
                        denom = 1.0 + abs(i-j) + abs(k-l) + abs(i-k) + abs(j-l)
                        ERI[i, j, k, l] = 0.1 / denom
        
        self.electron_repulsion = ERI
        return ERI
    
    def calculate_annihilation_integrals(self) -> np.ndarray:
        """
        Calculate electron-positron annihilation integrals.
        
        Only relevant for systems with antimatter.
        
        Returns:
            Annihilation integral tensor A
        """
        if not self.system.has_antimatter:
            return None
        
        # Placeholder - in a real implementation, this would compute accurate integrals
        n_basis = self.system.basis_set.size
        
        # Create a simplified annihilation integral matrix
        # In this model, annihilation amplitude is proportional to basis function overlap
        A = np.zeros((n_basis, n_basis))
        
        for i in range(n_basis):
            for j in range(n_basis):
                # Simple model: annihilation proportional to basis function overlap
                A[i, j] = np.exp(-(i-j)**2 / 5.0) * 0.05
        
        self.annihilation = A
        return A
    
    def get_core_hamiltonian(self) -> np.ndarray:
        """
        Get the core Hamiltonian matrix (H = T + V).
        
        Returns:
            Core Hamiltonian matrix H
        """
        if self.kinetic is None or self.nuclear is None:
            raise ValueError("Kinetic and nuclear integrals must be calculated first")
        
        return self.kinetic + self.nuclear
    
    def get_all_integrals(self) -> Dict[str, np.ndarray]:
        """
        Get all calculated integrals.
        
        Returns:
            Dictionary with all integral matrices
        """
        result = {
            'overlap': self.overlap,
            'kinetic': self.kinetic,
            'nuclear': self.nuclear,
            'electron_repulsion': self.electron_repulsion
        }
        
        if self.annihilation is not None:
            result['annihilation'] = self.annihilation
            
        return result