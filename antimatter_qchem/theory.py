"""
Core theoretical concepts for antimatter quantum chemistry simulations

The key differences between matter and antimatter in quantum chemistry:
1. Charge inversion: electrons (-) → positrons (+), protons (+) → antiprotons (-)
2. Modified Coulomb interactions
3. Potential annihilation interactions
4. Different relativistic effects due to different masses and interactions
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from enum import Enum

class ParticleType(Enum):
    """Enumeration for different particle types."""
    ELECTRON = 1
    POSITRON = 2
    PROTON = 3
    ANTIPROTON = 4
    NEUTRON = 5
    ANTINEUTRON = 6

class Particle:
    """Class representing a fundamental particle with properties."""
    
    def __init__(self, 
                 type_: ParticleType, 
                 mass: float, 
                 charge: float,
                 spin: float = 0.5):
        """
        Initialize a particle.
        
        Args:
            type_: The particle type
            mass: Mass of the particle in atomic units
            charge: Electric charge in elementary charge units
            spin: Spin quantum number (0.5 for fermions)
        """
        self.type = type_
        self.mass = mass
        self.charge = charge
        self.spin = spin
    
    @property
    def is_antimatter(self) -> bool:
        """Determine if this is an antimatter particle."""
        return self.type in [ParticleType.POSITRON, 
                            ParticleType.ANTIPROTON, 
                            ParticleType.ANTINEUTRON]
    
    def __str__(self) -> str:
        return f"{self.type.name}(mass={self.mass}, charge={self.charge}, spin={self.spin})"

# Standard particles with their properties in atomic units
STANDARD_PARTICLES = {
    ParticleType.ELECTRON: Particle(ParticleType.ELECTRON, 1.0, -1.0, 0.5),
    ParticleType.POSITRON: Particle(ParticleType.POSITRON, 1.0, 1.0, 0.5),
    ParticleType.PROTON: Particle(ParticleType.PROTON, 1836.15, 1.0, 0.5),
    ParticleType.ANTIPROTON: Particle(ParticleType.ANTIPROTON, 1836.15, -1.0, 0.5),
    ParticleType.NEUTRON: Particle(ParticleType.NEUTRON, 1838.68, 0.0, 0.5),
    ParticleType.ANTINEUTRON: Particle(ParticleType.ANTINEUTRON, 1838.68, 0.0, 0.5)
}

class InteractionType(Enum):
    """Types of interactions between particles."""
    COULOMB = 1         # Electrostatic interaction
    ANNIHILATION = 2    # Electron-positron annihilation
    NUCLEAR = 3         # Nuclear forces
    EXCHANGE = 4        # Quantum exchange interaction

def coulomb_potential(particle1: Particle, particle2: Particle, distance: float) -> float:
    """
    Calculate the Coulomb interaction potential between two particles.
    
    V(r) = q₁q₂/r
    
    Args:
        particle1: First particle
        particle2: Second particle
        distance: Distance between particles in Bohr
        
    Returns:
        Coulomb potential in Hartree
    """
    # Avoid division by zero
    if distance < 1e-10:
        return float('inf') if particle1.charge * particle2.charge > 0 else float('-inf')
    
    return particle1.charge * particle2.charge / distance

def annihilation_probability(electron: Particle, positron: Particle, distance: float) -> float:
    """
    Calculate probability of electron-positron annihilation at a given distance.
    
    The annihilation probability is modeled as a Gaussian function of distance:
    P(r) = exp(-r²/σ²)
    
    Args:
        electron: Electron particle
        positron: Positron particle
        distance: Distance between particles in Bohr
        
    Returns:
        Annihilation probability (0 to 1)
    """
    # Characteristic length for annihilation (~1 Bohr)
    sigma = 1.0
    
    # Verify we have electron and positron
    if electron.type != ParticleType.ELECTRON or positron.type != ParticleType.POSITRON:
        return 0.0
    
    # Calculate probability (Gaussian decay with distance)
    return np.exp(-(distance**2) / (sigma**2))

def calculate_exchange_integral(orbital1: np.ndarray, orbital2: np.ndarray, 
                              coulomb_operator: np.ndarray) -> float:
    """
    Calculate exchange integral between two orbitals.
    
    K_{ij} = ∫∫ φᵢ*(r₁)φⱼ(r₁) (1/r₁₂) φᵢ(r₂)φⱼ*(r₂) dr₁dr₂
    
    Args:
        orbital1: First orbital wavefunction values on a grid
        orbital2: Second orbital wavefunction values on a grid
        coulomb_operator: Matrix of 1/r₁₂ values on the grid
        
    Returns:
        Exchange integral value
    """
    # This is a simplified version - in reality this would involve a 4D integral
    # For a real implementation, sophisticated numerical integration would be needed
    return np.sum(np.outer(np.conj(orbital1) * orbital2, 
                          orbital1 * np.conj(orbital2)) * coulomb_operator)