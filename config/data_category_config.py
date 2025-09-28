"""
Configuration for data category filtering and evaluation.

This module defines the available data categories and provides utilities
for filtering evaluation data based on specific categories.
"""

from enum import Enum
from typing import Dict, List, Set, Optional
import os


class DataCategory(Enum):
    """Available data categories for evaluation."""
    MEASUREMENT = "measurement"
    PHYSICAL_METRIC = "physical_metric"
    RATIO_PERCENTAGE = "ratio_percentage"
    SIGNBOARD_AND_ICON = "signboard_and_icon"
    TEMPORAL = "temporal"
    OTHER = "other"


class DataSubCategory(Enum):
    """Available data subcategories for fine-grained evaluation."""
    # Measurement subcategories
    DISTANCE = "distance"
    LENGTH_AREA_VOLUME = "length_area_volume"
    
    # Physical metric subcategories
    SPEED = "speed"
    WEIGHT = "weight"
    
    # Ratio percentage subcategories
    GRAPH = "graph"
    STATISTICS = "statistics"
    
    # Signboard and icon subcategories
    GROUP = "group"
    PRICE = "price"
    
    # Temporal subcategories
    CALENDAR_AGE = "calendar_age"
    CLOCK = "clock"
    
    # Other subcategories
    COUNT = "count"
    DIALOGUE = "dialogue"
    LABEL = "label"


class DataCategoryConfig:
    """Configuration for data category filtering."""
    
    # Mapping from categories to their subcategories
    CATEGORY_SUBCATEGORY_MAP: Dict[DataCategory, List[DataSubCategory]] = {
        DataCategory.MEASUREMENT: [
            DataSubCategory.DISTANCE,
            DataSubCategory.LENGTH_AREA_VOLUME
        ],
        DataCategory.PHYSICAL_METRIC: [
            DataSubCategory.SPEED,
            DataSubCategory.WEIGHT
        ],
        DataCategory.RATIO_PERCENTAGE: [
            DataSubCategory.GRAPH,
            DataSubCategory.STATISTICS
        ],
        DataCategory.SIGNBOARD_AND_ICON: [
            DataSubCategory.GROUP,
            DataSubCategory.PRICE
        ],
        DataCategory.TEMPORAL: [
            DataSubCategory.CALENDAR_AGE,
            DataSubCategory.CLOCK
        ],
        DataCategory.OTHER: [
            DataSubCategory.COUNT,
            DataSubCategory.DIALOGUE,
            DataSubCategory.LABEL
        ]
    }
    
    # Mapping from subcategories to their file paths
    SUBCATEGORY_FILE_MAP: Dict[DataSubCategory, str] = {
        DataSubCategory.DISTANCE: "measurement/distance.json",
        DataSubCategory.LENGTH_AREA_VOLUME: "measurement/length_area_volume.json",
        DataSubCategory.SPEED: "physical_metric/speed.json",
        DataSubCategory.WEIGHT: "physical_metric/weight.json",
        DataSubCategory.GRAPH: "ratio_percentage/graph.json",
        DataSubCategory.STATISTICS: "ratio_percentage/statistics.json",
        DataSubCategory.GROUP: "signboard_and_icon/group.json",
        DataSubCategory.PRICE: "signboard_and_icon/price.json",
        DataSubCategory.CALENDAR_AGE: "temporal/calendar_age.json",
        DataSubCategory.CLOCK: "temporal/clock.json",
        DataSubCategory.COUNT: "other/count.json",
        DataSubCategory.DIALOGUE: "other/dialogue.json",
        DataSubCategory.LABEL: "other/label.json"
    }
    
    def __init__(
        self,
        enabled_categories: Optional[List[DataCategory]] = None,
        enabled_subcategories: Optional[List[DataSubCategory]] = None,
        metadata_dir: str = "data/metadata"
    ):
        """
        Initialize data category configuration.
        
        Args:
            enabled_categories: List of enabled categories. If None, all categories are enabled.
            enabled_subcategories: List of enabled subcategories. If None, all subcategories are enabled.
            metadata_dir: Path to the metadata directory containing category data files.
        """
        self.metadata_dir = metadata_dir
        
        # If categories are specified but subcategories are not, derive subcategories from categories
        if enabled_categories is not None and enabled_subcategories is None:
            self.enabled_categories = enabled_categories
            self.enabled_subcategories = []
            for category in enabled_categories:
                self.enabled_subcategories.extend(self.CATEGORY_SUBCATEGORY_MAP[category])
        # If subcategories are specified, derive categories from subcategories
        elif enabled_subcategories is not None:
            self.enabled_subcategories = enabled_subcategories
            # Determine which categories are needed based on subcategories
            needed_categories = set()
            for subcategory in enabled_subcategories:
                parent_category = self._get_parent_category_static(subcategory)
                needed_categories.add(parent_category)
            self.enabled_categories = list(needed_categories)
        # If neither is specified, enable all
        else:
            self.enabled_categories = list(DataCategory)
            self.enabled_subcategories = list(DataSubCategory)
        
        # Validate that enabled subcategories belong to enabled categories
        self._validate_configuration()
    
    def _validate_configuration(self):
        """Validate that enabled subcategories belong to enabled categories."""
        if self.enabled_subcategories and self.enabled_categories:
            valid_subcategories = set()
            for category in self.enabled_categories:
                valid_subcategories.update(self.CATEGORY_SUBCATEGORY_MAP[category])
            
            invalid_subcategories = set(self.enabled_subcategories) - valid_subcategories
            if invalid_subcategories:
                raise ValueError(
                    f"Invalid subcategories for enabled categories: {invalid_subcategories}"
                )
    
    def get_enabled_data_files(self) -> Dict[DataSubCategory, str]:
        """
        Get mapping of enabled subcategories to their data file paths.
        
        Returns:
            Dictionary mapping subcategories to absolute file paths.
        """
        enabled_files = {}
        
        for subcategory in self.enabled_subcategories:
            # Check if the subcategory's parent category is enabled
            parent_category = self._get_parent_category(subcategory)
            if parent_category in self.enabled_categories:
                relative_path = self.SUBCATEGORY_FILE_MAP[subcategory]
                absolute_path = os.path.join(self.metadata_dir, relative_path)
                enabled_files[subcategory] = absolute_path
        
        return enabled_files
    
    def _get_parent_category(self, subcategory: DataSubCategory) -> DataCategory:
        """Get the parent category for a given subcategory."""
        return self._get_parent_category_static(subcategory)
    
    @staticmethod
    def _get_parent_category_static(subcategory: DataSubCategory) -> DataCategory:
        """Get the parent category for a given subcategory (static version)."""
        for category, subcategories in DataCategoryConfig.CATEGORY_SUBCATEGORY_MAP.items():
            if subcategory in subcategories:
                return category
        raise ValueError(f"No parent category found for subcategory: {subcategory}")
    
    def get_question_ids_for_categories(self) -> Set[str]:
        """
        Get all question IDs for the enabled categories.
        
        Returns:
            Set of question IDs that belong to enabled categories.
        """
        # This method would need to load and parse the data files
        # For now, return an empty set as a placeholder
        return set()
    
    def to_dict(self) -> Dict[str, any]:
        """Convert configuration to dictionary representation."""
        return {
            "enabled_categories": [cat.value for cat in self.enabled_categories],
            "enabled_subcategories": [subcat.value for subcat in self.enabled_subcategories],
            "metadata_dir": self.metadata_dir,
            "enabled_data_files": {
                subcat.value: path 
                for subcat, path in self.get_enabled_data_files().items()
            }
        }


def create_category_config_from_names(
    category_names: Optional[List[str]] = None,
    subcategory_names: Optional[List[str]] = None,
    metadata_dir: str = "data/metadata"
) -> DataCategoryConfig:
    """
    Create a DataCategoryConfig from category and subcategory names.
    
    Args:
        category_names: List of category names (strings).
        subcategory_names: List of subcategory names (strings).
        metadata_dir: Path to metadata directory.
        
    Returns:
        Configured DataCategoryConfig instance.
        
    Raises:
        ValueError: If invalid category or subcategory names are provided.
    """
    enabled_categories = None
    enabled_subcategories = None
    
    if category_names:
        try:
            enabled_categories = [DataCategory(name) for name in category_names]
        except ValueError as e:
            valid_categories = [cat.value for cat in DataCategory]
            raise ValueError(
                f"Invalid category name in {category_names}. "
                f"Valid categories: {valid_categories}"
            ) from e
    
    if subcategory_names:
        try:
            enabled_subcategories = [DataSubCategory(name) for name in subcategory_names]
        except ValueError as e:
            valid_subcategories = [subcat.value for subcat in DataSubCategory]
            raise ValueError(
                f"Invalid subcategory name in {subcategory_names}. "
                f"Valid subcategories: {valid_subcategories}"
            ) from e
    
    return DataCategoryConfig(
        enabled_categories=enabled_categories,
        enabled_subcategories=enabled_subcategories,
        metadata_dir=metadata_dir
    )
