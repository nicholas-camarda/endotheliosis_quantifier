"""
Metadata Processor for Medical Image Analysis

This module provides utilities to process and standardize metadata files
for medical image analysis projects. It supports multiple input formats
and converts them to a standardized, project-agnostic structure.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union, cast

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class MetadataProcessor:
    """
    A flexible metadata processor for medical image analysis projects.
    
    Supports multiple input formats and converts them to standardized structures
    that are project-agnostic and production-ready.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the metadata processor.
        
        Args:
            config: Configuration dictionary with processing options
        """
        self.config = config or {}
        self.standardized_data = None
        
    def process_glomeruli_scoring_matrix(
        self, 
        file_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None
    ) -> pd.DataFrame:
        """
        Process a glomeruli scoring matrix Excel file into standardized format.
        
        Args:
            file_path: Path to the Excel file
            output_path: Optional path to save standardized CSV
            
        Returns:
            Standardized DataFrame with columns: [subject_id, glomerulus_id, score]
        """
        logger.info(f"Processing glomeruli scoring matrix: {file_path}")
        
        # Read the Excel file with proper header handling
        df = pd.read_excel(file_path, header=1)
        
        # Clean up the data
        df = self._clean_glomeruli_matrix(df)
        
        # Convert to long format (standardized)
        standardized_df = self._convert_to_long_format(df)
        
        # Save if output path provided
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            standardized_df.to_csv(output_path, index=False)
            logger.info(f"Saved standardized metadata to: {output_path}")
            
        self.standardized_data = standardized_df
        return standardized_df
    
    def _clean_glomeruli_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the raw glomeruli scoring matrix.
        
        Args:
            df: Raw DataFrame from Excel file
            
        Returns:
            Cleaned DataFrame
        """
        # Create a copy to avoid chained indexing warnings
        df = df.copy()
        
        # Remove summary rows (containing text like 'Score\\Slide', 'My Median Score', etc.)
        summary_indicators = ['Score\\Slide', 'My Median Score', 'My Average Score', 'Master Score']
        mask = ~df['Glomerulus #'].astype(str).isin(summary_indicators)
        df = df.loc[mask]
        
        # Convert glomerulus numbers to integers
        df.loc[:, 'Glomerulus #'] = pd.to_numeric(df['Glomerulus #'], errors='coerce')
        df = df.dropna(subset=['Glomerulus #'])
        
        # Clean up column names (remove unnamed columns and duplicates)
        valid_columns = ['Glomerulus #'] + [col for col in df.columns if not col.startswith('Unnamed') and col != 'Glomerulus #']
        df = df.loc[:, valid_columns]
        
        logger.info(f"Cleaned matrix shape: {df.shape}")
        logger.info(f"Valid columns: {len(valid_columns)}")
        
        return df
    
    def _convert_to_long_format(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert wide format (subjects as columns) to long format (standardized).
        
        Args:
            df: Wide format DataFrame
            
        Returns:
            Long format DataFrame with columns: [subject_id, glomerulus_id, score]
        """
        logger.info(f"Converting matrix with shape: {df.shape}")
        
        # Use pandas stack/unstack for robust wide-to-long conversion
        # Set glomerulus ID as index
        df_indexed = df.set_index('Glomerulus #')
        
        # Stack to convert to long format
        long_df = df_indexed.stack().reset_index()
        long_df.columns = ['glomerulus_id', 'subject_id', 'score']
        
        # Remove rows with missing scores
        long_df = long_df.dropna(subset=['score'])
        
        # Convert scores to numeric using modern pandas
        long_df['score'] = pd.to_numeric(long_df['score'], errors='coerce')
        long_df = long_df.dropna(subset=['score'])
        
        # Ensure proper data types using modern pandas - handle each column carefully
        long_df['glomerulus_id'] = pd.to_numeric(long_df['glomerulus_id'], errors='coerce').astype('Int64')  # type: ignore[attr-defined]
        long_df['score'] = pd.to_numeric(long_df['score'], errors='coerce').astype('float64')  # type: ignore[attr-defined]
        long_df['subject_id'] = long_df['subject_id'].astype('string')  # type: ignore[attr-defined]
        
        # Remove any remaining NaN values
        long_df = long_df.dropna()
        
        logger.info(f"Converted to long format: {long_df.shape}")
        return long_df
    
    def create_subject_summary(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Create a summary of subjects with their statistics.
        
        Args:
            df: Standardized DataFrame (uses self.standardized_data if None)
            
        Returns:
            Summary DataFrame with subject statistics
        """
        if df is None:
            df = self.standardized_data
            
        if df is None:
            raise ValueError("No data available. Run process_glomeruli_scoring_matrix first.")
        
        # Modern pandas aggregation syntax (ensure DataFrame return)
        grouped = df.groupby('subject_id', as_index=False).agg({
            'score': ['count', 'mean', 'std', 'min', 'max', 'median']
        }).round(3)
        summary_df: pd.DataFrame = cast(pd.DataFrame, grouped)
        
        # Flatten column names using modern pandas
        summary_df.columns = ['subject_id', 'num_glomeruli', 'mean_score', 'std_score', 'min_score', 'max_score', 'median_score']
        
        return summary_df
    
    def validate_data_quality(self, df: Optional[pd.DataFrame] = None, raw_data_dir: Optional[str] = None) -> Dict:
        """
        Complete validation of data quality including image/mask pairs and score coverage.
        
        Args:
            df: Standardized DataFrame (uses self.standardized_data if None)
            raw_data_dir: Path to raw data directory for file system validation
            
        Returns:
            Dictionary with comprehensive validation results
        """
        if df is None:
            df = self.standardized_data
            
        if df is None:
            raise ValueError("No data available. Run process_glomeruli_scoring_matrix first.")
        
        # Metadata quality validation
        # Ensure numeric types explicitly to satisfy static typing
        score_series: pd.Series = pd.to_numeric(df['score'], errors='coerce')  # type: ignore[assignment]
        score_np: np.ndarray = score_series.to_numpy(dtype=float, copy=False)
        min_score = float(np.nanmin(score_np)) if score_np.size else 0.0
        max_score = float(np.nanmax(score_np)) if score_np.size else 0.0
        unique_scores_list = [float(x) for x in sorted(np.unique(score_np[~np.isnan(score_np)]).tolist())]
        subjects_desc = df.groupby('subject_id').size().describe()
        subjects_stats = {str(k): float(v) for k, v in subjects_desc.to_dict().items()}
        glom_series: pd.Series = pd.to_numeric(df['glomerulus_id'], errors='coerce')  # type: ignore[assignment]
        glom_np: np.ndarray = glom_series.to_numpy(dtype=float, copy=False)
        s_max = float(np.nanmax(glom_np)) if glom_np.size else 0.0
        max_glom_int = int(s_max) if np.isfinite(s_max) and s_max > 0 else 0
        subj_n = int(df['subject_id'].nunique())
        completeness = float(len(df) / (subj_n * max_glom_int)) if max_glom_int > 0 else 0.0

        metadata_validation = {
            'total_subjects': int(df['subject_id'].nunique()),
            'total_glomeruli': int(len(df)),
            'score_range': [min_score, max_score],
            'unique_scores': unique_scores_list,
            'missing_scores': int(df['score'].isna().sum()),
            'subjects_with_data': subjects_stats,
            'data_completeness': completeness
        }
        
        validation_results = {
            'metadata_quality': metadata_validation
        }
        
        # File system validation if raw_data_dir provided
        if raw_data_dir:
            validation_results.update(self._validate_file_system(raw_data_dir, df))
        
        return validation_results
    
    def _validate_file_system(self, raw_data_dir: str, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate image/mask pairs and score coverage."""
        from pathlib import Path

        # raw_data_dir should be the images directory, so we need to find the masks directory
        images_dir = Path(raw_data_dir)
        if not images_dir.exists():
            return {'file_system_validation': {'error': 'Images directory not found'}}
        
        # Find the masks directory (should be at the same level as images)
        data_dir = images_dir.parent
        masks_dir = data_dir / "masks"
        
        if not masks_dir.exists():
            return {'file_system_validation': {'error': 'Masks directory not found'}}
        
        # Get all subject directories from images
        image_subject_dirs = [d for d in images_dir.iterdir() if d.is_dir() and d.name.startswith('T')]
        
        file_validation = {
            'total_subjects_found': len(image_subject_dirs),
            'subjects_with_images': 0,
            'subjects_with_masks': 0,
            'subjects_with_scores': 0,
            'image_mask_pairs': 0,
            'images_without_masks': 0,
            'images_without_scores': 0,
            'masks_without_images': 0,
            'scores_without_images': 0,
            'validation_errors': []
        }
        
        for subject_dir in image_subject_dirs:
            subject_id = subject_dir.name
            
            # Find images in this subject directory
            images = list(subject_dir.glob("*.png")) + list(subject_dir.glob("*.tif"))
            
            # Find corresponding masks in the masks directory
            mask_subject_dir = masks_dir / subject_id
            masks = []
            if mask_subject_dir.exists():
                masks = list(mask_subject_dir.glob("*.png")) + list(mask_subject_dir.glob("*.tif"))
            
            if images:
                file_validation['subjects_with_images'] += 1
            
            if masks:
                file_validation['subjects_with_masks'] += 1
            
            # Check for scores for this subject
            subject_scores = df[df['subject_id'].str.contains(subject_id, na=False)]
            if not subject_scores.empty:
                file_validation['subjects_with_scores'] += 1
            
            # Validate image/mask pairs and score coverage
            for image in images:
                image_stem = image.stem  # e.g., "T19_Image0"
                expected_mask = mask_subject_dir / f"{image_stem}_mask{image.suffix}"
                
                # Check if mask exists
                if expected_mask.exists():
                    file_validation['image_mask_pairs'] += 1
                else:
                    file_validation['images_without_masks'] += 1
                    file_validation['validation_errors'].append(f"No mask for {image}")
                
                # Check if this image has corresponding scores in metadata
                # Extract subject and image number from image name
                # T19_Image0 -> T19, Image0
                if '_Image' in image_stem:
                    subject_part = image_stem.split('_Image')[0]  # T19
                    image_part = image_stem.split('_Image')[1]    # 0
                    
                    # Look for scores for this subject in metadata
                    subject_scores = df[df['subject_id'].str.startswith(subject_part + '-')]
                    if subject_scores.empty:
                        file_validation['images_without_scores'] += 1
                        file_validation['validation_errors'].append(f"No scores for {image}")
            
            # Check for masks without images
            for mask in masks:
                mask_stem = mask.stem.replace('_mask', '')  # e.g., "T19_Image0"
                expected_image = subject_dir / f"{mask_stem}{mask.suffix}"
                
                if not expected_image.exists():
                    file_validation['masks_without_images'] += 1
                    file_validation['validation_errors'].append(f"No image for {mask}")
        
        # Calculate overall validation status
        total_issues = (file_validation['images_without_masks'] + 
                       file_validation['masks_without_images'] + 
                       file_validation['scores_without_images'])
        
        file_validation['overall_status'] = 'PASS' if total_issues == 0 else 'FAIL'
        file_validation['total_issues'] = total_issues
        
        return {'file_system_validation': file_validation}
    
    def export_for_ml_pipeline(self, output_dir: Union[str, Path]) -> Dict[str, Path]:
        """
        Export metadata in formats suitable for ML pipelines.
        
        Args:
            output_dir: Directory to save exported files
            
        Returns:
            Dictionary mapping file types to their paths
        """
        if self.standardized_data is None:
            raise ValueError("No data available. Run process_glomeruli_scoring_matrix first.")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        exported_files = {}
        
        # Export standardized CSV using modern pandas
        csv_path = output_dir / "standardized_metadata.csv"
        self.standardized_data.to_csv(csv_path, index=False)
        exported_files['standardized_csv'] = csv_path
        
        # Export subject summary
        summary = self.create_subject_summary()
        summary_path = output_dir / "subject_summary.csv"
        summary.to_csv(summary_path, index=False)
        exported_files['subject_summary'] = summary_path
        
        # Export validation report
        validation = self.validate_data_quality(raw_data_dir="raw_data/preeclampsia_project/data/images")
        validation_path = output_dir / "validation_report.json"
        with open(validation_path, 'w') as f:
            json.dump(validation, f, indent=2)
        exported_files['validation_report'] = validation_path
        
        # Export subject-to-score mapping for easy lookup using modern pandas
        score_mapping = self.standardized_data.set_index(['subject_id', 'glomerulus_id'])['score'].to_dict()
        
        # Convert tuple keys to strings for JSON serialization
        json_mapping = {}
        for key, value in score_mapping.items():
            if isinstance(key, tuple):
                json_key = f"{key[0]}_{key[1]}"
            else:
                json_key = str(key)
            json_mapping[json_key] = float(value)
            
        mapping_path = output_dir / "score_mapping.json"
        with open(mapping_path, 'w') as f:
            json.dump(json_mapping, f, indent=2)
        exported_files['score_mapping'] = mapping_path
        
        logger.info(f"Exported metadata files to: {output_dir}")
        return exported_files


def process_metadata_file(
    input_file: Union[str, Path],
    output_dir: Union[str, Path],
    file_type: str = "auto"
) -> Dict[str, Path]:
    """
    Convenience function to process a metadata file and export standardized formats.
    
    Args:
        input_file: Path to input metadata file
        output_dir: Directory to save processed files
        file_type: Type of input file ("auto", "glomeruli_matrix", "csv", "json")
        
    Returns:
        Dictionary mapping file types to their paths
    """
    processor = MetadataProcessor()
    
    if file_type == "auto":
        # Auto-detect file type using modern pathlib
        input_path = Path(input_file)
        if input_path.suffix.lower() == '.xlsx':
            file_type = "glomeruli_matrix"
        elif input_path.suffix.lower() == '.csv':
            file_type = "csv"
        elif input_path.suffix.lower() == '.json':
            file_type = "json"
    
    if file_type == "glomeruli_matrix":
        processor.process_glomeruli_scoring_matrix(input_file)
    else:
        raise ValueError(f"Unsupported file type: {file_type}")
    
    return processor.export_for_ml_pipeline(output_dir)


if __name__ == "__main__":
    # Example usage
    import logging
    # logging configured centrally via eq.utils.logger.setup_logging
    
    # Process the current metadata file
    input_file = "raw_data/preeclampsia_project/subject_metadata.xlsx"
    output_dir = "derived_data/glomeruli_data/metadata"
    
    if Path(input_file).exists():
        exported_files = process_metadata_file(input_file, output_dir)
        print("Exported files:")
        for file_type, path in exported_files.items():
            print(f"  {file_type}: {path}")
    else:
        print(f"Input file not found: {input_file}")
