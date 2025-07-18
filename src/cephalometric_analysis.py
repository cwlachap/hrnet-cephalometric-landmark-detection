#!/usr/bin/env python3
"""
Cephalometric Analysis Module
Calculates standard cephalometric measurements from 19 landmarks.
"""

import numpy as np
import math

class CephalometricAnalyzer:
    """
    Analyzer for cephalometric measurements using 19 landmarks.
    
    Landmark order (1-19):
    1. Sella turcica
    2. Nasion 
    3. Orbitale 
    4. Porion 
    5. Subspinale 
    6. Supramentale 
    7. Pogonion 
    8. Menton
    9. Gnathion
    10. Gonion 
    11. Incision inferius (Lower Incisor Tip)
    12. Incision superius (Upper Incisor Tip)
    13. Upper lip 
    14. Lower lip 
    15. Subnasale 
    16. Soft tissue pogonion 
    17. Posterior nasal spine 
    18. Anterior nasal spine 
    19. Articulare
    """
    
    def __init__(self, landmarks):
        """
        Initialize with landmark coordinates.
        
        Args:
            landmarks: numpy array of shape (19, 2) with (x, y) coordinates
        """
        self.landmarks = landmarks
        self.landmark_names = [
            "Sella", "Nasion", "Orbitale", "Porion", "Subspinale", "Supramentale",
            "Pogonion", "Menton", "Gnathion", "Gonion", "Lower Incisor Tip",
            "Upper Incisor Tip", "Upper Lip", "Lower Lip", "Subnasale", 
            "Soft Tissue Pogonion", "Posterior Nasal Spine", "Anterior Nasal Spine", "Articulare"
        ]
        
        # Define landmark indices for easier access
        self.S = 0   # Sella
        self.N = 1   # Nasion
        self.Or = 2  # Orbitale
        self.Po = 3  # Porion
        self.A = 4   # Subspinale
        self.B = 5   # Supramentale
        self.Pog = 6 # Pogonion
        self.Me = 7  # Menton
        self.Gn = 8  # Gnathion
        self.Go = 9  # Gonion
        self.L1 = 10 # Lower Incisor Tip
        self.U1 = 11 # Upper Incisor Tip
        self.UL = 12 # Upper Lip
        self.LL = 13 # Lower Lip
        self.Sn = 14 # Subnasale
        self.PoG = 15 # Soft Tissue Pogonion
        self.PNS = 16 # Posterior Nasal Spine
        self.ANS = 17 # Anterior Nasal Spine
        self.Ar = 18 # Articulare
    
    def calculate_angle(self, p1, p2, p3):
        """
        Calculate angle at p2 formed by points p1-p2-p3.
        
        Args:
            p1, p2, p3: Point indices or coordinate arrays
            
        Returns:
            Angle in degrees
        """
        if isinstance(p1, int):
            p1 = self.landmarks[p1]
        if isinstance(p2, int):
            p2 = self.landmarks[p2]
        if isinstance(p3, int):
            p3 = self.landmarks[p3]
            
        # Vectors from p2 to p1 and p2 to p3
        v1 = p1 - p2
        v2 = p3 - p2
        
        # Calculate angle using dot product
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        cos_angle = np.clip(cos_angle, -1, 1)  # Handle numerical errors
        angle_rad = np.arccos(cos_angle)
        angle_deg = np.degrees(angle_rad)
        
        return angle_deg
    
    def calculate_line_angle(self, p1, p2, reference_horizontal=True):
        """
        Calculate angle of line p1-p2 relative to horizontal or vertical.
        
        Args:
            p1, p2: Point indices or coordinate arrays
            reference_horizontal: If True, angle to horizontal; if False, to vertical
            
        Returns:
            Angle in degrees
        """
        if isinstance(p1, int):
            p1 = self.landmarks[p1]
        if isinstance(p2, int):
            p2 = self.landmarks[p2]
            
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        
        if reference_horizontal:
            angle_rad = np.arctan2(dy, dx)
        else:
            angle_rad = np.arctan2(dx, dy)
            
        angle_deg = np.degrees(angle_rad)
        
        # Normalize to 0-180 range
        angle_deg = abs(angle_deg)
        if angle_deg > 180:
            angle_deg = 360 - angle_deg
            
        return angle_deg
    
    def calculate_distance(self, p1, p2):
        """Calculate Euclidean distance between two points."""
        if isinstance(p1, int):
            p1 = self.landmarks[p1]
        if isinstance(p2, int):
            p2 = self.landmarks[p2]
            
        return np.linalg.norm(p2 - p1)
    
    def sna_angle(self):
        """Calculate SNA angle (Sella-Nasion-Subspinale)."""
        return self.calculate_angle(self.S, self.N, self.A)
    
    def snb_angle(self):
        """Calculate SNB angle (Sella-Nasion-Supramentale)."""
        return self.calculate_angle(self.S, self.N, self.B)
    
    def anb_angle(self):
        """Calculate ANB angle (difference between SNA and SNB)."""
        return self.sna_angle() - self.snb_angle()
    
    def fma_angle(self):
        """
        Calculate FMA (Frankfort-Mandibular Plane Angle).
        Frankfort plane: Orbitale to Porion
        Mandibular plane: Gonion to Menton
        """
        # Frankfort plane vector
        fp_vector = self.landmarks[self.Po] - self.landmarks[self.Or]
        # Mandibular plane vector
        mp_vector = self.landmarks[self.Me] - self.landmarks[self.Go]
        
        # Calculate angle between the two vectors
        cos_angle = np.dot(fp_vector, mp_vector) / (np.linalg.norm(fp_vector) * np.linalg.norm(mp_vector))
        cos_angle = np.clip(cos_angle, -1, 1)
        angle_rad = np.arccos(cos_angle)
        angle_deg = np.degrees(angle_rad)
        
        return angle_deg
    
    def sn_mp_angle(self):
        """
        Calculate SN-MP angle (Sella-Nasion to Mandibular Plane).
        """
        # SN plane vector
        sn_vector = self.landmarks[self.N] - self.landmarks[self.S]
        # Mandibular plane vector
        mp_vector = self.landmarks[self.Me] - self.landmarks[self.Go]
        
        # Calculate angle between the two vectors
        cos_angle = np.dot(sn_vector, mp_vector) / (np.linalg.norm(sn_vector) * np.linalg.norm(mp_vector))
        cos_angle = np.clip(cos_angle, -1, 1)
        angle_rad = np.arccos(cos_angle)
        angle_deg = np.degrees(angle_rad)
        
        return angle_deg
    
    def wits_appraisal(self):
        """
        Calculate Wits appraisal (approximated).
        This is a simplified version using the horizontal distance between A and B points.
        """
        # Horizontal distance between Subspinale and Supramentale
        distance = abs(self.landmarks[self.A][0] - self.landmarks[self.B][0])
        
        # Convert to mm (assuming pixel spacing of 0.1mm)
        distance_mm = distance * 0.1
        
        return distance_mm
    
    def pfh_afh_ratio(self):
        """
        Calculate PFH/AFH ratio.
        PFH (Posterior Facial Height): Sella to Gonion
        AFH (Anterior Facial Height): Nasion to Menton
        """
        pfh = self.calculate_distance(self.S, self.Go)
        afh = self.calculate_distance(self.N, self.Me)
        
        if afh == 0:
            return 0
        
        return pfh / afh
    
    def u1_sn_angle(self):
        """
        Calculate U1-SN angle (Upper incisor to Sella-Nasion plane).
        """
        # Use upper incisor tip and root to define incisor long axis
        # Since we only have tip, we'll approximate using tip and ANS
        u1_vector = self.landmarks[self.U1] - self.landmarks[self.ANS]
        sn_vector = self.landmarks[self.N] - self.landmarks[self.S]
        
        # Calculate angle between the two vectors
        cos_angle = np.dot(u1_vector, sn_vector) / (np.linalg.norm(u1_vector) * np.linalg.norm(sn_vector))
        cos_angle = np.clip(cos_angle, -1, 1)
        angle_rad = np.arccos(cos_angle)
        angle_deg = np.degrees(angle_rad)
        
        return angle_deg
    
    def impa_angle(self):
        """
        Calculate IMPA (Incisor Mandibular Plane Angle).
        Angle between lower incisor and mandibular plane.
        """
        # Lower incisor vector (approximated using tip and supramentale)
        l1_vector = self.landmarks[self.L1] - self.landmarks[self.B]
        # Mandibular plane vector
        mp_vector = self.landmarks[self.Me] - self.landmarks[self.Go]
        
        # Calculate angle between the two vectors
        cos_angle = np.dot(l1_vector, mp_vector) / (np.linalg.norm(l1_vector) * np.linalg.norm(mp_vector))
        cos_angle = np.clip(cos_angle, -1, 1)
        angle_rad = np.arccos(cos_angle)
        angle_deg = np.degrees(angle_rad)
        
        return angle_deg
    
    def interincisal_angle(self):
        """
        Calculate interincisal angle (U1-L1).
        Angle between upper and lower incisors.
        """
        # Upper incisor vector
        u1_vector = self.landmarks[self.U1] - self.landmarks[self.ANS]
        # Lower incisor vector
        l1_vector = self.landmarks[self.L1] - self.landmarks[self.B]
        
        # Calculate angle between the two vectors
        cos_angle = np.dot(u1_vector, l1_vector) / (np.linalg.norm(u1_vector) * np.linalg.norm(l1_vector))
        cos_angle = np.clip(cos_angle, -1, 1)
        angle_rad = np.arccos(cos_angle)
        angle_deg = np.degrees(angle_rad)
        
        return angle_deg
    
    def calculate_all_measurements(self):
        """
        Calculate all cephalometric measurements.
        
        Returns:
            Dictionary with all measurements
        """
        measurements = {
            'SNA': self.sna_angle(),
            'SNB': self.snb_angle(),
            'ANB': self.anb_angle(),
            'FMA': self.fma_angle(),
            'SN-MP': self.sn_mp_angle(),
            'Wits': self.wits_appraisal(),
            'PFH/AFH': self.pfh_afh_ratio(),
            'U1-SN': self.u1_sn_angle(),
            'IMPA': self.impa_angle(),
            'Interincisal': self.interincisal_angle()
        }
        
        return measurements
    
    def get_normal_ranges(self):
        """
        Get normal ranges for cephalometric measurements.
        
        Returns:
            Dictionary with normal ranges for each measurement
        """
        normal_ranges = {
            'SNA': (80, 84),      # degrees
            'SNB': (78, 82),      # degrees  
            'ANB': (0, 4),        # degrees
            'FMA': (20, 30),      # degrees
            'SN-MP': (28, 38),    # degrees
            'Wits': (-1, 3),      # mm (for males)
            'PFH/AFH': (0.60, 0.80),  # ratio
            'U1-SN': (100, 110),  # degrees
            'IMPA': (88, 95),     # degrees
            'Interincisal': (120, 140)  # degrees
        }
        
        return normal_ranges
    
    def interpret_measurement(self, measurement_name, value):
        """
        Interpret a measurement value relative to normal ranges.
        
        Args:
            measurement_name: Name of the measurement
            value: Measured value
            
        Returns:
            Tuple of (status, interpretation)
        """
        normal_ranges = self.get_normal_ranges()
        
        if measurement_name not in normal_ranges:
            return "Unknown", "No reference range available"
        
        min_normal, max_normal = normal_ranges[measurement_name]
        
        if min_normal <= value <= max_normal:
            return "Normal", "Within normal range"
        elif value < min_normal:
            return "Low", f"Below normal range ({min_normal}-{max_normal})"
        else:
            return "High", f"Above normal range ({min_normal}-{max_normal})" 