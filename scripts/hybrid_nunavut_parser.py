import pandas as pd
import re
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HybridNunavutParser:
    """
    Hybrid parser combining the best features of both FixedNunavutParser and ImprovedNunavutParser.
    
    Key features:
    1. ✅ FIXED interruption handling from FixedNunavutParser (continues speech after >>Applause)
    2. ✅ Comprehensive speaker patterns from ImprovedNunavutParser
    3. ✅ Procedural filtering from ImprovedNunavutParser
    4. ✅ Better date extraction heuristics from FixedNunavutParser
    """
    
    def __init__(self):
        # 🔥 COMPREHENSIVE speaker patterns from ImprovedNunavutParser
        self.speaker_patterns = [
            # Standard format: Mr./Ms./Hon. Name: speech
            r'^((?:Mr\.|Ms\.|Hon\.|Mrs\.|Dr\.)\s+[^:()]+?):\s*(.+)$',
            # Speaker role patterns
            r'^(Speaker(?:\s+\([^)]+\))?)\s*:\s*(.+)$',
            r'^((?:Chairman|Chairperson|Deputy Speaker|Premier|Minister|Clerk)(?:\s+\([^)]+\))?)\s*:\s*(.+)$',
            # Interpretation patterns
            r'^([^:]+\s+\(interpretation\))\s*:\s*(.+)$',
            # Name only patterns (last resort)
            r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s*:\s*(.+)$'
        ]
        
        # Date patterns for extraction
        self.date_patterns = [
            r'([A-Z]+DAY,?\s+[A-Z]+\s+\d{1,2},?\s+\d{4})',  # THURSDAY, APRIL 1, 1999
            r'([A-Z]+\s+[A-Z]+\s+\d{1,2},?\s+\d{4})',       # MAY 12, 1999  
            r'(\d{4}-\d{2}-\d{2})',                          # 1999-04-01
            r'([A-Z]+\s+\d{1,2},?\s+\d{4})'                  # APRIL 1, 1999
        ]
        
        # 🔥 ENHANCED skip patterns from ImprovedNunavutParser
        self.skip_patterns = [
            r'^LEGISLATIVE ASSEMBLY',
            r'^\d+(?:st|nd|rd|th)\s+Session',
            r'^\d+(?:st|nd|rd|th)\s+Assembly',
            r'^HANSARD$',
            r'^Official Report$',
            r'^Members of the Legislative Assembly$',
            r'^Officers$',
            r'^Clerk$',
            r'^Deputy Clerk\s*$',
            r'^Table of Contents',
            r'^Page\s*\d*$',
            r'^DAILY REFERENCES',
            r'^[A-Z]\.\s*Page',
            r'^ITEM \d+:',
            r'^Motion \d+',
            r'^Bill \d+',
            r'^Tabled Document',
            r'^\s*$',  # Empty lines
            r'^--+$',   # Just dashes
            r'^\d+\s*$',  # Just numbers
            r'^Mr\.\s+[A-Z][a-z]+\s+[A-Z][a-z]+$',  # Just name without colon
            r'^Ms\.\s+[A-Z][a-z]+\s+[A-Z][a-z]+$',  # Just name without colon
            r'^Hon\.\s+[A-Z][a-z]+\s+[A-Z][a-z]+$',  # Just name without colon
            r'^\([A-Z][a-z]+\)$',  # Just constituency names
        ]
        
        # 🔥 INTERRUPTION patterns for proper handling
        self.interruption_patterns = [
            r'^>>Applause\s*$',
            r'^>>Laughter\s*$', 
            r'^--Applause\s*$',
            r'^--Laughter\s*$',
            r'^Some Hon\. Members:',
            r'^All Hon\. Members:',
            r'^House adjourned',
            r'^The House resumed',
            r'^Motion carried',
            r'^Motion defeated',
            r'^Question put',
            r'^Division',
            r'^\(Applause\)$',
            r'^\(Laughter\)$'
        ]
        
        # 🔥 PROCEDURAL phrases from ImprovedNunavutParser
        self.procedural_phrases = [
            r'^Thank you\.?\s*$',
            r'^Are you agreed\??\s*$',
            r'^Agreed\.?\s*$',
            r'^Question\.?\s*$',
            r'^Motion carried\.?\s*$',
            r'^All in favour\??\s*$',
            r'^All opposed\??\s*$',
            r'^I move\.?\s*$',
            r'^I second\.?\s*$'
        ]
    
    def extract_date(self, line: str) -> Optional[str]:
        """
        🔥 IMPROVED date extraction with heuristics from FixedNunavutParser.
        Prevents extracting dates from within speeches.
        """
        clean_line = line.strip()

        # Heuristic: A real date header is unlikely to be a long sentence or end
        # with punctuation typical of a speech
        if len(clean_line) > 60 or clean_line.endswith(('.', '?', '!', '."')):
            return None

        for pattern in self.date_patterns:
            match = re.search(pattern, line, re.IGNORECASE)
            if match:
                # Heuristic: The line should not be significantly longer than the
                # date string itself. This prevents capturing dates from within a
                # full sentence.
                date_text = match.group(1)
                if len(clean_line) > len(date_text) + 20:
                    continue # This is likely a sentence, not a header.

                # If the heuristics pass, proceed with the original parsing logic.
                date_str = match.group(1)
                try:
                    if ',' in date_str:
                        date_parts = date_str.replace(',', '').split()
                        if date_parts[0].upper() in ['MONDAY', 'TUESDAY', 'WEDNESDAY', 'THURSDAY', 'FRIDAY', 'SATURDAY', 'SUNDAY']:
                            date_parts = date_parts[1:]
                        
                        if len(date_parts) >= 3:
                            month_day_year = date_parts[:3]
                            try:
                                return datetime.strptime(' '.join(month_day_year), '%B %d %Y').strftime('%Y-%m-%d')
                            except ValueError:
                                try:
                                    return datetime.strptime(' '.join(month_day_year), '%b %d %Y').strftime('%Y-%m-%d')
                                except ValueError:
                                    pass
                    else:
                        return datetime.strptime(date_str, '%Y-%m-%d').strftime('%Y-%m-%d')
                except ValueError:
                    continue
        return None
    
    def should_skip_line(self, line: str) -> bool:
        """Determine if a line should be skipped entirely."""
        line = line.strip()
        if not line:
            return True
            
        for pattern in self.skip_patterns:
            if re.match(pattern, line, re.IGNORECASE):
                return True
        return False
    
    def is_interruption(self, line: str) -> bool:
        """🔥 CRITICAL: Check if line is a procedural interruption."""
        for pattern in self.interruption_patterns:
            if re.match(pattern, line, re.IGNORECASE):
                return True
        return False
    
    def is_procedural_phrase(self, text: str) -> bool:
        """Check if text is just a short procedural phrase."""
        text = text.strip()
        for pattern in self.procedural_phrases:
            if re.match(pattern, text, re.IGNORECASE):
                return True
        return False
    
    def extract_speaker_and_speech(self, line: str) -> Optional[Tuple[str, str]]:
        """🔥 ENHANCED: Extract speaker and speech from a line with procedural filtering."""
        for pattern in self.speaker_patterns:
            match = re.match(pattern, line)
            if match:
                speaker = match.group(1).strip()
                speech = match.group(2).strip()
                
                # Filter out very short procedural speeches
                if len(speech) < 5 or self.is_procedural_phrase(speech):
                    return None
                    
                return speaker, speech
        return None
    
    def is_speech_continuation(self, line: str, current_speech: str) -> bool:
        """
        🔥 FIXED: Determine if a line is a continuation of current speech.
        Key fix: Interruptions are handled separately in main loop.
        """
        line = line.strip()
        
        # Skip if it's a clear administrative line
        if self.should_skip_line(line):
            return False
            
        # Skip if it looks like a new speaker (contains colon)
        if ':' in line and self.extract_speaker_and_speech(line):
            return False
            
        # Skip topic/section headers
        if re.match(r'^\d+\s*-\s*\d+\s*\(\d+\):', line):
            return False
            
        # ✅ FIXED: Don't check for interruptions here!
        # Interruptions should be handled separately in main loop
        
        # If line has meaningful content and we're in a speech, it's likely a continuation
        if len(line) > 3 and current_speech:
            return True
            
        return False
    
    def parse_hansard(self, file_path: str) -> pd.DataFrame:
        """
        🔥 HYBRID parser with the best of both worlds:
        - FixedNunavutParser's interruption handling
        - ImprovedNunavutParser's comprehensive patterns and filtering
        """
        logger.info(f"Starting hybrid parser on {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
        except Exception as e:
            logger.error(f"Error reading file: {e}")
            return pd.DataFrame()
        
        lines = content.split('\n')
        speeches = []
        current_date = None
        current_speech = ""
        current_speaker = ""
        in_speech = False
        speech_counter = 0
        
        logger.info(f"Processing {len(lines)} lines...")
        
        for i, line in enumerate(lines):
            line = line.strip()
            
            if not line:
                continue
            
            # Extract date
            extracted_date = self.extract_date(line)
            if extracted_date:
                current_date = extracted_date
                logger.debug(f"Found date: {current_date}")
                continue
            
            # Skip administrative lines
            if self.should_skip_line(line):
                continue
            
            # 🔥 CRITICAL FIX: Handle interruptions BEFORE checking continuations
            if self.is_interruption(line):
                # Skip interruption but STAY in speech mode
                logger.debug(f"Skipping interruption: {line}")
                continue
            
            # Check for new speaker
            speaker_speech = self.extract_speaker_and_speech(line)
            if speaker_speech:
                # Save previous speech
                if in_speech and current_speech.strip() and current_date and current_speaker:
                    speeches.append({
                        'basepk': speech_counter,
                        'hid': f'nu.{current_date}.{speech_counter}',
                        'speechdate': current_date,
                        'speechtext': current_speech.strip(),
                        'speakername': current_speaker.strip()
                    })
                    speech_counter += 1
                
                # Start new speech
                current_speaker, current_speech = speaker_speech
                in_speech = True
                continue
            
            # 🔥 FIXED: Check speech continuation (interruptions already handled above)
            if in_speech and self.is_speech_continuation(line, current_speech):
                current_speech += " " + line
                continue
            
            # If we reach here and we're in a speech, something unexpected happened
            # End the current speech and continue
            if in_speech and current_speech.strip() and current_date and current_speaker:
                speeches.append({
                    'basepk': speech_counter,
                    'hid': f'nu.{current_date}.{speech_counter}',
                    'speechdate': current_date,
                    'speechtext': current_speech.strip(),
                    'speakername': current_speaker.strip()
                })
                speech_counter += 1
            
            in_speech = False
            current_speech = ""
            current_speaker = ""
        
        # Save last speech
        if in_speech and current_speech.strip() and current_date and current_speaker:
            speeches.append({
                'basepk': speech_counter,
                'hid': f'nu.{current_date}.{speech_counter}',
                'speechdate': current_date,
                'speechtext': current_speech.strip(),
                'speakername': current_speaker.strip()
            })
        
        logger.info(f"Extracted {len(speeches)} speeches")
        
        # Create DataFrame and clean
        df = pd.DataFrame(speeches)
        
        if df.empty:
            logger.warning("No speeches extracted!")
            return df
        
        # Additional cleaning
        logger.info("Performing additional cleaning...")
        
        # Convert speechdate to datetime
        df['speechdate'] = pd.to_datetime(df['speechdate'], errors='coerce')
        
        # Remove rows with invalid dates
        df = df.dropna(subset=['speechdate'])
        
        # Remove very short speeches (likely procedural)
        original_count = len(df)
        df = df[df['speechtext'].str.len() >= 15]
        logger.info(f"Removed {original_count - len(df)} very short speeches")
        
        # Remove duplicate speeches
        original_count = len(df)
        df = df.drop_duplicates(subset=['speechtext'], keep='first')
        logger.info(f"Removed {original_count - len(df)} duplicate speeches")
        
        # Sort by date and reset index
        df = df.sort_values('speechdate').reset_index(drop=True)
        
        # Update basepk to be sequential
        df['basepk'] = range(len(df))
        df['hid'] = df.apply(lambda row: f"nu.{row['speechdate'].strftime('%Y.%m.%d')}.{row['basepk']}", axis=1)
        
        logger.info(f"Final dataset contains {len(df)} speeches")
        if not df.empty:
            logger.info(f"Date range: {df['speechdate'].min()} to {df['speechdate'].max()}")
            logger.info(f"Average speech length: {df['speechtext'].str.len().mean():.1f} characters")
            logger.info(f"Unique speakers: {df['speakername'].nunique()}")
        
        return df

def main():
    """Test the hybrid parser."""
    parser = HybridNunavutParser()
    
    # Parse the data
    input_file = '../data/preprocessed_nunavut_hansard.txt'
    df_nunavut = parser.parse_hansard(input_file)
    
    if df_nunavut.empty:
        logger.error("No data extracted!")
        return
    
    # Save to CSV
    output_file = '../data/hybrid_nunavut_hansard.csv'
    df_nunavut.to_csv(output_file, index=False)
    logger.info(f"Hybrid parsed data saved to: {output_file}")
    
    # Display sample
    print("\nSample of hybrid parsed data:")
    print(df_nunavut[['speechdate', 'speakername', 'speechtext']].head())
    
    # Show statistics
    print(f"\nHybrid Dataset Statistics:")
    print(f"Total speeches: {len(df_nunavut)}")
    print(f"Unique speakers: {df_nunavut['speakername'].nunique()}")
    print(f"Date range: {df_nunavut['speechdate'].dt.date.min()} to {df_nunavut['speechdate'].dt.date.max()}")
    print(f"Average speech length: {df_nunavut['speechtext'].str.len().mean():.1f} characters")
    print(f"Median speech length: {df_nunavut['speechtext'].str.len().median():.1f} characters")

if __name__ == "__main__":
    main()