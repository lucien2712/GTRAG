"""
gtrag simplified time handling utilities.

This module provides time handling focused on YYYY-MM format standardization,
simplifying time operations and ensuring consistent format usage.
"""

import re
from datetime import datetime, date
from typing import List, Optional, Set, Tuple, Union, Dict, Any
from dataclasses import dataclass


@dataclass
class TimePoint:
    """Represents a standardized time point in YYYY-MM format."""
    value: str  # Standardized YYYY-MM value
    raw_value: str  # Original input value
    sort_key: str  # For sorting purposes (same as value for YYYY-MM)
    
    def __str__(self):
        return self.value
    
    def __lt__(self, other):
        return self.sort_key < other.sort_key
    
    def __eq__(self, other):
        return self.value == other.value


class TimeHandler:
    """
    Simplified time handler focused on YYYY-MM format standardization.
    
    Standard format: YYYY-MM (e.g., 2024-03, 2024-12)
    
    This ensures consistent time representation across the entire system
    and eliminates format conversion complexity.
    """
    
    # Pattern for YYYY-MM format validation
    YYYY_MM_PATTERN = re.compile(r'^(\d{4})-(\d{2})$')  # 2024-03
    
    @classmethod
    def parse_time(cls, time_value: Union[str, datetime, date]) -> Optional[TimePoint]:
        """
        Parse time value into standardized YYYY-MM format TimePoint.
        
        Args:
            time_value: Time specification in YYYY-MM format (e.g., "2024-03")
            
        Returns:
            TimePoint object or None if parsing fails
        """
        if not time_value:
            return None
        
        # Handle datetime objects - convert to YYYY-MM
        if isinstance(time_value, (datetime, date)):
            time_str = time_value.strftime('%Y-%m')
            return cls._create_timepoint(time_str)
        
        if not isinstance(time_value, str):
            return None
        
        time_str = time_value.strip()
        
        # Validate YYYY-MM format
        match = cls.YYYY_MM_PATTERN.match(time_str)
        if match:
            return cls._create_timepoint(time_str)
        
        # Invalid format
        return None
    
    @classmethod
    def _create_timepoint(cls, yyyy_mm: str) -> TimePoint:
        """Create a TimePoint for YYYY-MM format."""
        return TimePoint(
            value=yyyy_mm,
            raw_value=yyyy_mm,
            sort_key=yyyy_mm  # YYYY-MM is naturally sortable
        )
    
    
    
    @classmethod
    def extract_time_from_metadata(cls, metadata: Dict[str, Any]) -> Optional[TimePoint]:
        """
        Extract time information from metadata dictionary.
        Only uses the unified 'date' field.
        """
        if 'date' in metadata:
            return cls.parse_time(metadata['date'])
        
        return None
    
    @classmethod
    def calculate_time_similarity(cls, time1: TimePoint, time2: TimePoint, 
                                 max_distance: int = 4) -> float:
        """
        Calculate similarity between two YYYY-MM time points.
        
        Args:
            time1, time2: TimePoint objects to compare
            max_distance: Maximum month distance for similarity calculation
            
        Returns:
            Similarity score between 0.0 and 1.0
        """
        if time1.value == time2.value:
            return 1.0
        
        # Calculate month distance between YYYY-MM values
        try:
            year1, month1 = map(int, time1.value.split('-'))
            year2, month2 = map(int, time2.value.split('-'))
            
            # Convert to months since epoch for distance calculation
            months1 = year1 * 12 + month1
            months2 = year2 * 12 + month2
            distance = abs(months1 - months2)
            
            # Similarity decreases with month distance
            if distance <= max_distance:
                return 1.0 - distance / (max_distance + 1)
            else:
                return 0.0
                
        except (ValueError, TypeError):
            return 0.0
    
    @classmethod
    def is_adjacent(cls, time1: TimePoint, time2: TimePoint) -> bool:
        """Check if two YYYY-MM time points are adjacent (consecutive months)."""
        try:
            year1, month1 = map(int, time1.value.split('-'))
            year2, month2 = map(int, time2.value.split('-'))
            
            # Convert to months since epoch
            months1 = year1 * 12 + month1
            months2 = year2 * 12 + month2
            
            # Adjacent means difference of exactly 1 month
            return abs(months1 - months2) == 1
            
        except (ValueError, TypeError):
            return False
    
    @classmethod
    def expand_time_range(cls, time_range: List[str], expansion_mode: str = "strict") -> Set[str]:
        """
        Expand time range specification with different modes.
        
        Args:
            time_range: List of time specifications
            expansion_mode: "strict", "adjacent", or "expanded"
            
        Returns:
            Set of expanded time values
        """
        if not time_range:
            return set()
        
        # Parse all time points
        time_points = []
        for time_spec in time_range:
            parsed = cls.parse_time(time_spec)
            if parsed:
                time_points.append(parsed)
        
        if not time_points:
            return set()
        
        result = {tp.value for tp in time_points}
        
        if expansion_mode == "strict":
            return result
        
        elif expansion_mode == "adjacent":
            # Add adjacent time points
            for tp in time_points:
                adjacent_times = cls._get_adjacent_times(tp)
                result.update(adjacent_times)
        
        elif expansion_mode == "expanded":
            # Add broader range based on time spans
            result.update(cls._get_expanded_range(time_points))
        
        return result
    
    @classmethod
    def _get_adjacent_times(cls, time_point: TimePoint) -> Set[str]:
        """Get adjacent YYYY-MM time points (previous and next months)."""
        adjacent = set()
        
        try:
            year, month = map(int, time_point.value.split('-'))
            
            # Previous month
            if month == 1:
                adjacent.add(f"{year-1}-12")
            else:
                adjacent.add(f"{year}-{month-1:02d}")
            
            # Next month
            if month == 12:
                adjacent.add(f"{year+1}-01")
            else:
                adjacent.add(f"{year}-{month+1:02d}")
                
        except (ValueError, TypeError):
            pass
        
        return adjacent
    
    @classmethod
    def _get_expanded_range(cls, time_points: List[TimePoint]) -> Set[str]:
        """Get expanded YYYY-MM range covering the span of time points."""
        if not time_points:
            return set()
        
        # Sort time points
        sorted_points = sorted(time_points)
        start_point = sorted_points[0]
        end_point = sorted_points[-1]
        
        expanded = set()
        
        try:
            start_year, start_month = map(int, start_point.value.split('-'))
            end_year, end_month = map(int, end_point.value.split('-'))
            
            # Fill in all months between start and end
            current_year, current_month = start_year, start_month
            
            while (current_year < end_year or 
                   (current_year == end_year and current_month <= end_month)):
                expanded.add(f"{current_year}-{current_month:02d}")
                
                # Move to next month
                current_month += 1
                if current_month > 12:
                    current_month = 1
                    current_year += 1
                    
        except (ValueError, TypeError):
            pass
        
        return expanded