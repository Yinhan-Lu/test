import pandas as pd
import re
from typing import List, Dict, Tuple, Set
import logging
from collections import defaultdict, Counter
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HansardValidator:
    """
    Comprehensive validation system for Hansard speech extraction.
    
    This validator ensures:
    1. All actual speeches are extracted (completeness)
    2. No non-speech content is incorrectly classified as speech (accuracy)
    3. Speech attribution is correct
    4. Temporal consistency is maintained
    """
    
    def __init__(self):
        # Patterns to identify actual speech content in raw text
        self.speech_indicators = [
            r'^((?:Mr\.|Ms\.|Hon\.|Mrs\.|Dr\.)\s+[^:()]+?):\s*(.{20,})',  # Long speeches
            r'^(Speaker(?:\s+\([^)]+\))?)\s*:\s*(.{20,})',
            r'^((?:Chairman|Chairperson|Deputy Speaker|Premier|Minister|Clerk)(?:\s+\([^)]+\))?)\s*:\s*(.{20,})',
            r'^([^:]+\s+\(interpretation\))\s*:\s*(.{20,})',
        ]
        
        # Patterns that definitely indicate non-speech content
        self.non_speech_patterns = [
            r'^LEGISLATIVE ASSEMBLY',
            r'^\d+(?:st|nd|rd|th)\s+Session',
            r'^HANSARD$',
            r'^Official Report$',
            r'^Members of the Legislative Assembly$',
            r'^Officers$',
            r'^Table of Contents',
            r'^Page\s*\d*$',
            r'^\d+\s*-\s*\d+\s*\(\d+\):',  # Topic headers
            r'^>>Applause\s*$',
            r'^--Applause\s*$',
            r'^\(Applause\)$',
            r'^Motion carried',
            r'^Question put',
            r'^House adjourned',
        ]
        
        # Common procedural phrases that should not be standalone speeches
        self.procedural_only = [
            r'^Thank you\.?\s*$',
            r'^Question\.?\s*$',
            r'^Are you agreed\??\s*$',
            r'^Agreed\.?\s*$',
            r'^Motion carried\.?\s*$',
        ]
    
    def extract_speakers_from_raw(self, raw_text_path: str) -> Dict[str, List[Tuple[str, str]]]:
        """
        Extract all speaker instances from raw text for completeness validation.
        
        Returns:
            Dict mapping speaker names to list of (date, speech_snippet) tuples
        """
        logger.info("Extracting speakers from raw text for validation...")
        
        with open(raw_text_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        lines = content.split('\n')
        speakers_found = defaultdict(list)
        current_date = None
        
        # Date patterns
        date_patterns = [
            r'([A-Z]+DAY,?\s+[A-Z]+\s+\d{1,2},?\s+\d{4})',
            r'([A-Z]+\s+[A-Z]+\s+\d{1,2},?\s+\d{4})',
            r'(\d{4}-\d{2}-\d{2})',
            r'([A-Z]+\s+\d{1,2},?\s+\d{4})'
        ]
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Extract date
            for pattern in date_patterns:
                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    current_date = match.group(1)
                    break
            
            # Look for speeches
            for pattern in self.speech_indicators:
                match = re.match(pattern, line)
                if match:
                    speaker = match.group(1).strip()
                    speech_snippet = match.group(2)[:100] + "..."  # First 100 chars
                    speakers_found[speaker].append((current_date, speech_snippet))
        
        return dict(speakers_found)
    
    def validate_completeness(self, raw_text_path: str, extracted_df: pd.DataFrame) -> Dict[str, any]:
        """
        Validate that all speech content was extracted from the raw text.
        """
        logger.info("Validating extraction completeness...")
        
        # Get speakers from raw text
        raw_speakers = self.extract_speakers_from_raw(raw_text_path)
        
        # Get speakers from extracted data
        extracted_speakers = defaultdict(list)
        for _, row in extracted_df.iterrows():
            speaker = row['speakername']
            date = row['speechdate'].strftime('%Y-%m-%d') if pd.notna(row['speechdate']) else 'Unknown'
            speech_snippet = row['speechtext'][:100] + "..."
            extracted_speakers[speaker].append((date, speech_snippet))
        
        # Compare counts
        raw_total = sum(len(speeches) for speeches in raw_speakers.values())
        extracted_total = len(extracted_df)
        
        extraction_rate = (extracted_total / raw_total * 100) if raw_total > 0 else 0
        
        # Find missing speakers
        missing_speakers = set(raw_speakers.keys()) - set(extracted_speakers.keys())
        
        # Find speakers with significantly fewer speeches
        under_extracted = {}
        for speaker in raw_speakers:
            if speaker in extracted_speakers:
                raw_count = len(raw_speakers[speaker])
                extracted_count = len(extracted_speakers[speaker])
                if extracted_count < raw_count * 0.7:  # Less than 70% extracted
                    under_extracted[speaker] = {
                        'raw_count': raw_count,
                        'extracted_count': extracted_count,
                        'rate': extracted_count / raw_count * 100
                    }
        
        return {
            'raw_speech_count': raw_total,
            'extracted_speech_count': extracted_total,
            'extraction_rate': extraction_rate,
            'missing_speakers': list(missing_speakers),
            'under_extracted_speakers': under_extracted,
            'total_raw_speakers': len(raw_speakers),
            'total_extracted_speakers': len(extracted_speakers)
        }
    
    def validate_accuracy(self, extracted_df: pd.DataFrame) -> Dict[str, any]:
        """
        Validate that extracted content is actually speech content (not metadata/administrative).
        """
        logger.info("Validating extraction accuracy...")
        
        issues = []
        
        for i, row in extracted_df.iterrows():
            speech_text = row['speechtext']
            speaker = row['speakername']
            
            # Check if speech text matches non-speech patterns
            for pattern in self.non_speech_patterns:
                if re.match(pattern, speech_text, re.IGNORECASE):
                    issues.append({
                        'row': i,
                        'issue': 'Non-speech content',
                        'pattern': pattern,
                        'text': speech_text[:100] + "...",
                        'speaker': speaker
                    })
            
            # Check if speech is just a procedural phrase
            for pattern in self.procedural_only:
                if re.match(pattern, speech_text.strip(), re.IGNORECASE):
                    issues.append({
                        'row': i,
                        'issue': 'Procedural-only speech',
                        'pattern': pattern,
                        'text': speech_text,
                        'speaker': speaker
                    })
            
            # Check for very short speeches (likely procedural)
            if len(speech_text.strip()) < 10:
                issues.append({
                    'row': i,
                    'issue': 'Very short speech',
                    'text': speech_text,
                    'speaker': speaker
                })
            
            # Check for topic headers mistaken as speeches
            if re.match(r'^\d+\s*-\s*\d+\s*\(\d+\):', speech_text):
                issues.append({
                    'row': i,
                    'issue': 'Topic header as speech',
                    'text': speech_text,
                    'speaker': speaker
                })
        
        accuracy_rate = ((len(extracted_df) - len(issues)) / len(extracted_df) * 100) if len(extracted_df) > 0 else 100
        
        return {
            'total_speeches': len(extracted_df),
            'problematic_speeches': len(issues),
            'accuracy_rate': accuracy_rate,
            'issues': issues[:50]  # Limit to first 50 issues for review
        }
    
    def validate_speaker_consistency(self, extracted_df: pd.DataFrame) -> Dict[str, any]:
        """
        Validate speaker name consistency and identify potential attribution errors.
        """
        logger.info("Validating speaker consistency...")
        
        speaker_variants = defaultdict(list)
        speaker_stats = {}
        
        # Group similar speaker names
        for speaker in extracted_df['speakername'].unique():
            base_name = self._normalize_speaker_name(speaker)
            speaker_variants[base_name].append(speaker)
        
        # Identify potential issues
        inconsistent_speakers = {}
        for base_name, variants in speaker_variants.items():
            if len(variants) > 1:
                inconsistent_speakers[base_name] = variants
        
        # Speaker statistics
        speaker_counts = extracted_df['speakername'].value_counts()
        for speaker, count in speaker_counts.items():
            speeches = extracted_df[extracted_df['speakername'] == speaker]['speechtext']
            speaker_stats[speaker] = {
                'speech_count': count,
                'avg_length': speeches.str.len().mean(),
                'total_chars': speeches.str.len().sum()
            }
        
        return {
            'unique_speakers': len(extracted_df['speakername'].unique()),
            'speaker_variants': dict(inconsistent_speakers),
            'top_speakers': dict(speaker_counts.head(10)),
            'speaker_statistics': {k: v for k, v in list(speaker_stats.items())[:20]}  # Top 20
        }
    
    def _normalize_speaker_name(self, name: str) -> str:
        """Normalize speaker name for consistency checking."""
        # Remove titles and parenthetical information
        name = re.sub(r'^(Mr\.|Ms\.|Hon\.|Mrs\.|Dr\.)\s*', '', name)
        name = re.sub(r'\s*\([^)]+\)\s*', '', name)
        return name.strip().lower()
    
    def validate_temporal_consistency(self, extracted_df: pd.DataFrame) -> Dict[str, any]:
        """
        Validate temporal consistency of the extracted data.
        """
        logger.info("Validating temporal consistency...")
        
        if extracted_df.empty:
            return {'error': 'Empty dataset'}
        
        # Convert dates
        dates = pd.to_datetime(extracted_df['speechdate'], errors='coerce')
        valid_dates = dates.dropna()
        
        date_issues = []
        
        # Check for invalid dates
        invalid_dates = len(dates) - len(valid_dates)
        if invalid_dates > 0:
            date_issues.append(f"{invalid_dates} rows with invalid dates")
        
        # Check date range
        if len(valid_dates) > 0:
            min_date = valid_dates.min()
            max_date = valid_dates.max()
            date_span = (max_date - min_date).days
        else:
            min_date = max_date = date_span = None
            date_issues.append("No valid dates found")
        
        # Check for date gaps
        if len(valid_dates) > 1:
            unique_dates = sorted(valid_dates.dt.date.unique())
            gaps = []
            for i in range(1, len(unique_dates)):
                gap = (unique_dates[i] - unique_dates[i-1]).days
                if gap > 30:  # More than 30 days
                    gaps.append((unique_dates[i-1], unique_dates[i], gap))
        
        return {
            'total_records': len(extracted_df),
            'valid_dates': len(valid_dates),
            'invalid_dates': invalid_dates,
            'date_range': (min_date, max_date),
            'date_span_days': date_span,
            'unique_dates': len(valid_dates.dt.date.unique()) if len(valid_dates) > 0 else 0,
            'date_issues': date_issues,
            'large_gaps': gaps[:10] if 'gaps' in locals() else []
        }
    
    def run_comprehensive_validation(self, raw_text_path: str, extracted_csv_path: str) -> Dict[str, any]:
        """
        Run all validation tests and return comprehensive report.
        """
        logger.info("Starting comprehensive validation...")
        
        # Load extracted data
        try:
            extracted_df = pd.read_csv(extracted_csv_path)
            logger.info(f"Loaded {len(extracted_df)} extracted speeches")
        except Exception as e:
            logger.error(f"Error loading extracted data: {e}")
            return {'error': f'Could not load extracted data: {e}'}
        
        # Run all validations
        validation_results = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'input_files': {
                'raw_text': raw_text_path,
                'extracted_csv': extracted_csv_path
            }
        }
        
        try:
            validation_results['completeness'] = self.validate_completeness(raw_text_path, extracted_df)
        except Exception as e:
            logger.error(f"Completeness validation failed: {e}")
            validation_results['completeness'] = {'error': str(e)}
        
        try:
            validation_results['accuracy'] = self.validate_accuracy(extracted_df)
        except Exception as e:
            logger.error(f"Accuracy validation failed: {e}")
            validation_results['accuracy'] = {'error': str(e)}
        
        try:
            validation_results['speaker_consistency'] = self.validate_speaker_consistency(extracted_df)
        except Exception as e:
            logger.error(f"Speaker consistency validation failed: {e}")
            validation_results['speaker_consistency'] = {'error': str(e)}
        
        try:
            validation_results['temporal_consistency'] = self.validate_temporal_consistency(extracted_df)
        except Exception as e:
            logger.error(f"Temporal consistency validation failed: {e}")
            validation_results['temporal_consistency'] = {'error': str(e)}
        
        # Calculate overall score
        scores = []
        if 'completeness' in validation_results and 'extraction_rate' in validation_results['completeness']:
            scores.append(validation_results['completeness']['extraction_rate'])
        if 'accuracy' in validation_results and 'accuracy_rate' in validation_results['accuracy']:
            scores.append(validation_results['accuracy']['accuracy_rate'])
        
        if scores:
            validation_results['overall_score'] = np.mean(scores)
        else:
            validation_results['overall_score'] = 0
        
        return validation_results
    
    def generate_report(self, validation_results: Dict[str, any]) -> str:
        """Generate a human-readable validation report."""
        
        report = []
        report.append("=" * 60)
        report.append("HANSARD EXTRACTION VALIDATION REPORT")
        report.append("=" * 60)
        report.append(f"Generated: {validation_results.get('timestamp', 'Unknown')}")
        report.append(f"Overall Score: {validation_results.get('overall_score', 0):.1f}%")
        report.append("")
        
        # Completeness
        if 'completeness' in validation_results:
            comp = validation_results['completeness']
            report.append("COMPLETENESS ANALYSIS:")
            report.append(f"  - Raw speeches found: {comp.get('raw_speech_count', 'N/A')}")
            report.append(f"  - Speeches extracted: {comp.get('extracted_speech_count', 'N/A')}")
            report.append(f"  - Extraction rate: {comp.get('extraction_rate', 0):.1f}%")
            report.append(f"  - Missing speakers: {len(comp.get('missing_speakers', []))}")
            report.append(f"  - Under-extracted speakers: {len(comp.get('under_extracted_speakers', {}))}")
            report.append("")
        
        # Accuracy
        if 'accuracy' in validation_results:
            acc = validation_results['accuracy']
            report.append("ACCURACY ANALYSIS:")
            report.append(f"  - Total speeches: {acc.get('total_speeches', 'N/A')}")
            report.append(f"  - Problematic speeches: {acc.get('problematic_speeches', 'N/A')}")
            report.append(f"  - Accuracy rate: {acc.get('accuracy_rate', 0):.1f}%")
            report.append("")
        
        # Speaker Consistency
        if 'speaker_consistency' in validation_results:
            sc = validation_results['speaker_consistency']
            report.append("SPEAKER CONSISTENCY:")
            report.append(f"  - Unique speakers: {sc.get('unique_speakers', 'N/A')}")
            report.append(f"  - Speaker variants detected: {len(sc.get('speaker_variants', {}))}")
            report.append("")
        
        # Temporal Consistency
        if 'temporal_consistency' in validation_results:
            tc = validation_results['temporal_consistency']
            report.append("TEMPORAL CONSISTENCY:")
            report.append(f"  - Total records: {tc.get('total_records', 'N/A')}")
            report.append(f"  - Valid dates: {tc.get('valid_dates', 'N/A')}")
            report.append(f"  - Date range: {tc.get('date_range', 'N/A')}")
            report.append(f"  - Unique dates: {tc.get('unique_dates', 'N/A')}")
            report.append("")
        
        return "\n".join(report)

def main():
    """Main function to run validation."""
    validator = HansardValidator()
    
    # Validate improved parser results
    raw_text_path = '../data/preprocessed_nunavut_hansard.txt'
    extracted_csv_path = '../data/improved_nunavut_hansard.csv'
    
    # Run validation
    results = validator.run_comprehensive_validation(raw_text_path, extracted_csv_path)
    
    # Generate and display report
    report = validator.generate_report(results)
    print(report)
    
    # Save detailed results
    import json
    with open('../data/validation_report_improved.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info("Validation complete. Detailed results saved to validation_report_improved.json")

if __name__ == "__main__":
    main() 