"""
gtrag flexible time handling utilities.

This module provides a unified interface for handling various time formats
and temporal operations, replacing the quarter-only limitation with flexible
time support for different use cases.
"""

import re
from datetime import datetime, date
from typing import List, Optional, Set, Tuple, Union, Dict, Any
from dataclasses import dataclass
from enum import Enum


class TimeGranularity(Enum):
    """Supported time granularities."""
    DAY = "day"
    MONTH = "month"
    QUARTER = "quarter"
    YEAR = "year"
    CUSTOM = "custom"


@dataclass
class TimePoint:
    """Represents a standardized time point with granularity information."""
    value: str  # Standardized time value
    granularity: TimeGranularity
    raw_value: str  # Original input value
    sort_key: str  # For sorting purposes
    
    def __str__(self):
        return self.value
    
    def __lt__(self, other):
        return self.sort_key < other.sort_key
    
    def __eq__(self, other):
        return self.value == other.value


class TimeHandler:
    """
    Unified time handler for parsing, standardizing, and comparing various time formats.
    
    Supports:
    - ISO dates: 2024-03-15, 2024-03, 2024
    - Quarter format: 2024Q1, 2023Q4
    - Month names: March 2024, Mar 2024
    - Custom labels: Sprint-1, Phase-A, etc.
    """
    
    # Regex patterns for different time formats
    PATTERNS = {
        'iso_date': re.compile(r'^(\d{4})-(\d{2})-(\d{2})$'),          # 2024-03-15
        'iso_month': re.compile(r'^(\d{4})-(\d{2})$'),                 # 2024-03
        'iso_year': re.compile(r'^(\d{4})$'),                          # 2024
        'quarter': re.compile(r'^(\d{4})Q([1-4])$'),                   # 2024Q1
        'month_year': re.compile(r'^(\w+)\s+(\d{4})$'),                # March 2024
        'month_year_short': re.compile(r'^(\w{3})\s+(\d{4})$'),        # Mar 2024
        'custom_label': re.compile(r'^[A-Za-z][A-Za-z0-9_-]*$'),       # Sprint-1, Phase-A
    }
    
    # Month name mappings
    MONTH_NAMES = {
        'january': '01', 'jan': '01', 'february': '02', 'feb': '02',
        'march': '03', 'mar': '03', 'april': '04', 'apr': '04',
        'may': '05', 'june': '06', 'jun': '06', 'july': '07', 'jul': '07',
        'august': '08', 'aug': '08', 'september': '09', 'sep': '09',
        'october': '10', 'oct': '10', 'november': '11', 'nov': '11',
        'december': '12', 'dec': '12'
    }
    
    @classmethod
    def parse_time(cls, time_value: Union[str, datetime, date]) -> Optional[TimePoint]:
        """
        Parse various time formats into a standardized TimePoint.
        
        Args:
            time_value: Time specification in various formats
            
        Returns:
            TimePoint object or None if parsing fails
        """
        if not time_value:
            return None
        
        # Handle datetime objects
        if isinstance(time_value, (datetime, date)):
            time_str = time_value.strftime('%Y-%m-%d')
            return cls._parse_iso_date(time_str, time_str)
        
        if not isinstance(time_value, str):
            return None
        
        time_str = time_value.strip()
        
        # Try different patterns
        for pattern_name, pattern in cls.PATTERNS.items():
            match = pattern.match(time_str)
            if match:
                return cls._parse_by_pattern(pattern_name, match, time_str)
        
        # If no pattern matches, treat as custom label
        return TimePoint(
            value=time_str,
            granularity=TimeGranularity.CUSTOM,
            raw_value=time_str,
            sort_key=f"custom_{time_str}"
        )
    
    @classmethod
    def _parse_by_pattern(cls, pattern_name: str, match: re.Match, raw_value: str) -> TimePoint:
        """Parse based on specific pattern match."""
        if pattern_name == 'iso_date':
            year, month, day = match.groups()
            return TimePoint(
                value=f"{year}-{month}-{day}",
                granularity=TimeGranularity.DAY,
                raw_value=raw_value,
                sort_key=f"{year}{month}{day}"
            )
        
        elif pattern_name == 'iso_month':
            year, month = match.groups()
            return TimePoint(
                value=f"{year}-{month}",
                granularity=TimeGranularity.MONTH,
                raw_value=raw_value,
                sort_key=f"{year}{month}00"
            )
        
        elif pattern_name == 'iso_year':
            year = match.groups()[0]
            return TimePoint(
                value=year,
                granularity=TimeGranularity.YEAR,
                raw_value=raw_value,
                sort_key=f"{year}0000"
            )
        
        elif pattern_name == 'quarter':
            year, quarter = match.groups()
            return TimePoint(
                value=f"{year}Q{quarter}",
                granularity=TimeGranularity.QUARTER,
                raw_value=raw_value,
                sort_key=f"{year}{int(quarter):02d}00"
            )
        
        elif pattern_name in ['month_year', 'month_year_short']:
            month_name, year = match.groups()
            month_num = cls.MONTH_NAMES.get(month_name.lower())
            if month_num:
                return TimePoint(
                    value=f"{year}-{month_num}",
                    granularity=TimeGranularity.MONTH,
                    raw_value=raw_value,
                    sort_key=f"{year}{month_num}00"
                )
        
        elif pattern_name == 'custom_label':
            return TimePoint(
                value=raw_value,
                granularity=TimeGranularity.CUSTOM,
                raw_value=raw_value,
                sort_key=f"custom_{raw_value}"
            )
        
        return None
    
    @classmethod
    def _parse_iso_date(cls, time_str: str, raw_value: str) -> TimePoint:
        """Helper for parsing ISO date from datetime objects."""
        parts = time_str.split('-')
        year, month, day = parts[0], parts[1], parts[2]
        return TimePoint(
            value=f"{year}-{month}-{day}",
            granularity=TimeGranularity.DAY,
            raw_value=raw_value,
            sort_key=f"{year}{month}{day}"
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
        Calculate similarity between two time points.
        
        Args:
            time1, time2: TimePoint objects to compare
            max_distance: Maximum distance for similarity calculation
            
        Returns:
            Similarity score between 0.0 and 1.0
        """
        if time1.granularity != time2.granularity:
            # Different granularities get lower base similarity
            base_similarity = 0.3
        else:
            base_similarity = 0.7
        
        # For custom labels, use exact match only
        if time1.granularity == TimeGranularity.CUSTOM:
            return 1.0 if time1.value == time2.value else 0.0
        
        # Calculate sort key distance
        try:
            key1_num = int(time1.sort_key.replace('custom_', '0'))
            key2_num = int(time2.sort_key.replace('custom_', '0'))
            distance = abs(key1_num - key2_num)
        except (ValueError, TypeError):
            # Fallback for non-numeric sort keys
            return 1.0 if time1.value == time2.value else 0.0
        
        if distance == 0:
            return 1.0
        
        # Similarity decreases with distance
        if distance <= max_distance:
            return base_similarity * (1.0 - distance / (max_distance + 1))
        else:
            return 0.0
    
    @classmethod
    def is_adjacent(cls, time1: TimePoint, time2: TimePoint) -> bool:
        """Check if two time points are adjacent (consecutive)."""
        if time1.granularity != time2.granularity:
            return False
        
        if time1.granularity == TimeGranularity.CUSTOM:
            return False  # Custom labels don't have inherent ordering
        
        try:
            key1_num = int(time1.sort_key)
            key2_num = int(time2.sort_key)
            
            if time1.granularity == TimeGranularity.QUARTER:
                # For quarters, adjacent means consecutive quarters
                return abs(key1_num - key2_num) == 100  # Q1->Q2, Q2->Q3, etc.
            elif time1.granularity == TimeGranularity.MONTH:
                # For months, adjacent means consecutive months
                return abs(key1_num - key2_num) == 100
            elif time1.granularity == TimeGranularity.YEAR:
                # For years, adjacent means consecutive years
                return abs(key1_num - key2_num) == 10000
            elif time1.granularity == TimeGranularity.DAY:
                # For days, this is more complex - simplified version
                return abs(key1_num - key2_num) <= 1
                
        except (ValueError, TypeError):
            return False
        
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
        """Get adjacent time points for a given time point."""
        adjacent = set()
        
        if time_point.granularity == TimeGranularity.QUARTER:
            # Add previous and next quarters
            year, quarter = time_point.value.split('Q')
            year = int(year)
            quarter = int(quarter)
            
            # Previous quarter
            if quarter == 1:
                adjacent.add(f"{year-1}Q4")
            else:
                adjacent.add(f"{year}Q{quarter-1}")
            
            # Next quarter
            if quarter == 4:
                adjacent.add(f"{year+1}Q1")
            else:
                adjacent.add(f"{year}Q{quarter+1}")
        
        elif time_point.granularity == TimeGranularity.MONTH:
            # Add previous and next months
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
        
        elif time_point.granularity == TimeGranularity.YEAR:
            # Add previous and next years
            year = int(time_point.value)
            adjacent.add(str(year - 1))
            adjacent.add(str(year + 1))
        
        return adjacent
    
    @classmethod
    def _get_expanded_range(cls, time_points: List[TimePoint]) -> Set[str]:
        """Get expanded range covering the span of time points."""
        if not time_points:
            return set()
        
        # Sort time points
        sorted_points = sorted(time_points)
        start_point = sorted_points[0]
        end_point = sorted_points[-1]
        
        expanded = set()
        
        # For quarters, fill in the range
        if start_point.granularity == TimeGranularity.QUARTER and end_point.granularity == TimeGranularity.QUARTER:
            start_year, start_q = map(int, start_point.value.split('Q'))
            end_year, end_q = map(int, end_point.value.split('Q'))
            
            for year in range(start_year, end_year + 1):
                start_quarter = start_q if year == start_year else 1
                end_quarter = end_q if year == end_year else 4
                
                for quarter in range(start_quarter, end_quarter + 1):
                    expanded.add(f"{year}Q{quarter}")
        
        # Similar logic for other granularities...
        # (Implementation can be extended as needed)
        
        return expanded