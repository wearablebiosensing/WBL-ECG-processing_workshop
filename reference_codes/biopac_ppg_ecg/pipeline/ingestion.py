import os
import logging
import pandas as pd
import numpy as np
import time
from datetime import datetime
from typing import Tuple, Optional, Callable

class DataIngestion:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def load_biopac(self, file_path: str, progress_callback: Optional[Callable[[str, float], None]] = None) -> Tuple[pd.DataFrame, float, int]:
        """
        Loads Biopac data efficiently with progress tracking.
        Refined to avoid 'Error tokenizing data'.
        """
        self.logger.info(f"Loading Biopac file: {file_path}")
        
        if progress_callback: progress_callback("Parsing Header...", 0.0)
        
        # 1. Parse Header
        try:
            with open(file_path, 'r', encoding='utf-8-sig') as f:
                header_lines = [f.readline() for _ in range(100)]
        except UnicodeDecodeError:
             with open(file_path, 'r', encoding='latin-1') as f:
                header_lines = [f.readline() for _ in range(100)]
                
        sample_rate_ms = None
        start_dt = None
        data_start_line = None
        header_line = ""
        
        for i, line in enumerate(header_lines):
            l_lower = line.lower()
            if "msec/sample" in l_lower:
                try: sample_rate_ms = float(line.split()[0])
                except: pass
            if "recording on:" in l_lower:
                try:
                    t_str = line.split(":", 1)[1].strip()
                    for fmt in ["%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"]:
                        try:
                            start_dt = datetime.strptime(t_str, fmt)
                            break
                        except: continue
                except: pass
            if l_lower.startswith("sec,ch") or l_lower.startswith("sec, ch"):
                data_start_line = i
                header_line = line
                # Only break if we found everything? 
                # Actually we need all 3.
                
        if sample_rate_ms is None: raise ValueError("Header Error: 'msec/sample' not found. Is this a valid AcqKnowledge text export?")
        if start_dt is None: raise ValueError("Header Error: 'Recording on:' not found.")
        if data_start_line is None: raise ValueError("Header Error: 'sec,ch' not found.")
        
        # --- Robustness Fix (Phase 22) ---
        print(f"DEBUG: Header detected at line {data_start_line}: {header_line.strip()}")
        
        # Scan lines AFTER the header to skip metadata like "21347161 samples"
        # We look for the first line that starts with a digit/number AND isn't metadata
        actual_data_line = data_start_line + 1
        for i in range(data_start_line + 1, min(len(header_lines), data_start_line + 20)):
            line = header_lines[i].strip()
            if not line: continue
            
            l_lower = line.lower()
            if "samples" in l_lower: 
                print(f"DEBUG: Skipping metadata line {i}: {line}")
                continue # Skip sample count lines
            
            # Check if line starts with a digit
            check_line = line.lstrip(',')
            if check_line and (check_line[0].isdigit() or check_line.startswith('-') or check_line.startswith('.')):
                actual_data_line = i
                print(f"DEBUG: Data start verified at line {i}: {line[:50]}...")
                break
        
        skip_count = actual_data_line
        print(f"DEBUG: Final skip_count for read_csv: {skip_count}")
        
        fs = 1000.0 / sample_rate_ms
        start_ts_ms = int(start_dt.timestamp() * 1000)
        
        # Map Columns
        header_parts = [h.strip().upper() for h in header_line.strip().split(',')]
        col_indices = {}
        required_signals = {'CH2': 'ECG', 'CH3': 'PPG', 'CH7': 'SKT', 'CH16': 'EDA'}
        
        # Invert the map: found 'CH2' at idx 1 -> col_indices['ECG'] = 1
        for idx, col in enumerate(header_parts):
            for ch_code, ch_name in required_signals.items():
                if ch_code in col:
                    col_indices[ch_name] = idx
        
        if 'ECG' not in col_indices or 'PPG' not in col_indices:
            self.logger.warning(f"ECG/PPG not found in header columns: {header_parts}. Will try to fuzzy match.")
            
        # 2. Read Data
        if progress_callback: progress_callback("Reading Data...", 0.05)
        
        # Use simple read_csv first (no chunking) to debug if chunking caused the parser error?
        # No, user wants progress.
        # "Error tokenizing data. C error: Expected 1 fields in line 4, saw 3"
        # Since we use header=None, pandas expects consistent columns.
        # If line 4 (which is in the skipped region? No, skiprows handles lines before start)
        # Ah, skiprows=data_start_line + 1.
        # If the file has empty lines at the end or weird footer?
        # Try engine='python' which is more robust, but slower.
        # Or, explicit delimiter. Biopac Text export is usually CSV (',') or Tab.
        # 'sec,ch' implies comma.
        
        chunk_size = 50000 
        file_size = os.path.getsize(file_path)
        estimated_rows = file_size / 50 
        chunks = []
        rows_read = 0
        t0 = time.time()
        
        # Proactive Robustness: If we found metadata lines (skip_count > data_start_line + 1), it's a messy file.
        # Use Python engine immediately to avoid C-engine header inference issues.
        use_python_engine = (skip_count > (data_start_line + 1))
        
        # Explicit Column Names to prevent "Expected X fields" error
        # Construct names from header and fill remainder with dummies if needed
        # We rely on header_parts which we parsed earlier
        col_names = header_parts + [f"Extra_{i}" for i in range(10)] # Add buffer columns just in case
        
        csv_kwargs = dict(
            header=None,
            names=None, # Let it infer? No. If we pass names, we MUST match count.
            # actually better: just skip bad lines and let it read.
            # But "Expected 1 fields" means it mis-inferred.
            # So passing `names` fixes it?
            # Let's try passing `names` equal to the max columns we expect.
            # Actually, simply `engine='python'` is safer for ragged files.
            skiprows=skip_count,
            on_bad_lines='skip', 
            encoding='utf-8-sig',
            low_memory=False,
            sep=',', 
            dtype=str
        )
        
        # if use_python_engine:
        #      print("DEBUG: Messy file detected. Proactively using Python engine.")
        #      csv_kwargs['engine'] = 'python'
        #      csv_kwargs.pop('low_memory', None)
        
        chunks = []
        rows_read = 0
        reader = None
        
        # Helper to run the reading loop
        def read_loop(rdr, is_robust=False):
            nonlocal rows_read, chunks
            for j, chunk in enumerate(rdr):
                chunks.append(chunk)
                rows_read += len(chunk)
                if progress_callback and j % 5 == 0:
                    pct = min(rows_read / estimated_rows, 0.95)
                    lbl = "Reading (Robust)..." if is_robust else "Reading..."
                    progress_callback(f"{lbl} {rows_read:,} lines", pct)

        try:
            # Try C engine first
            try:
                reader = pd.read_csv(file_path, chunksize=chunk_size, **csv_kwargs)
            except UnicodeDecodeError:
                csv_kwargs['encoding'] = 'latin-1'
                reader = pd.read_csv(file_path, chunksize=chunk_size, **csv_kwargs)
            
            read_loop(reader)
                    
        except Exception as e:
            # Catch ParserError or other read errors (like the line 4 issue)
            print(f"DEBUG: C engine failed ({e}). Switching to Python engine.")
            self.logger.warning(f"C engine failed ({e}). Retrying with python engine.")
            
            if progress_callback: progress_callback("Retrying with robust parser...", 0.1)
            
            # Switch settings for Python engine
            csv_kwargs.pop('low_memory', None)
            csv_kwargs['engine'] = 'python'
            csv_kwargs['sep'] = ',' # explicit
            
            # Re-init chunks
            chunks = []
            rows_read = 0
            
            try:
                reader = pd.read_csv(file_path, chunksize=chunk_size, **csv_kwargs)
                read_loop(reader, is_robust=True)
            except Exception as e2:
                # If even robust fails, raise original or new
                raise ValueError(f"Failed to parse file even with robust engine: {e2}")

        df = pd.concat(chunks, ignore_index=True)
        
        if progress_callback: progress_callback("Processing Columns...", 0.98)
        
        # Rename based on indices
        rename_map = {}
        for name, idx in col_indices.items():
            if idx in df.columns:
                 rename_map[idx] = name
        
        df = df.rename(columns=rename_map)
        
        # Convert numeric (robustly)
        # Note: 'CH2' might read as object if there are "mv" strings or similar.
        # We enforce numeric.
        cols_to_keep = [c for c in ['ECG','PPG','SKT','EDA'] if c in df.columns]
        
        # Check if column 0 (Time) exists and is numeric
        # Assuming col 0 corresponds to 'sec'
        if 0 in df.columns:
             # Convert to numeric, errors=coerce turns bad values to NaN
             df[0] = pd.to_numeric(df[0], errors='coerce')
             df = df.dropna(subset=[0]) # Drop rows where time is invalid
        
        for c in cols_to_keep:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        
        df = df[cols_to_keep]
        
        # Fill NaNs? Linear interpolate small gaps
        df = df.interpolate(limit=10)
        df = df.fillna(method='bfill').fillna(method='ffill')

        # Timestamps
        n = len(df)
        time_vec = start_ts_ms + (np.arange(n) * sample_rate_ms)
        df['timestamp_ms'] = time_vec.astype('int64')
        
        return df, fs, start_ts_ms

    def load_labels_file(self, file_path: str) -> pd.DataFrame:
        # Same as before
        def try_read(fpath):
            if fpath.endswith('.xlsx') or fpath.endswith('.xls'):
                return pd.read_excel(fpath, header=None)
            try: return pd.read_csv(fpath, header=None, encoding='utf-8-sig')
            except: return pd.read_csv(fpath, header=None, encoding='latin-1')

        df_raw = try_read(file_path)
        valid_data = []
        
        # Check if header exists in first row (try to heuristic detect)
        has_header = False
        try:
             # If first value is "Timestamp" (string)
             if str(df_raw.iloc[0, 0]).lower() == "timestamp":
                 df_raw = df_raw.iloc[1:] # Skip header
                 has_header = True
        except: pass
        
        for idx, row in df_raw.iterrows():
            try:
                # User Format: 0=Timestamp, 1=Event Description
                ts = float(str(row[0]).strip())
                lbl = str(row[1]).strip() if len(row) > 1 else ""
                
                # If lbl is just "Event Description" (header row that wasn't caught), skip
                if lbl.lower() == "event description": continue
                
                valid_data.append({'timestamp_ms': int(ts), 'label': lbl})
            except: continue
        
        label_df = pd.DataFrame(valid_data)
        if label_df.empty: return pd.DataFrame(columns=['timestamp_ms', 'label'])
        return label_df.sort_values('timestamp_ms')

    def merge_biopac_labels(self, biopac_df: pd.DataFrame, label_df: pd.DataFrame, tolerance_ms: int = None) -> pd.DataFrame:
        if label_df.empty: return biopac_df
        biopac_df = biopac_df.sort_values('timestamp_ms')
        label_df = label_df.sort_values('timestamp_ms')
        
        # Phase 30 Fix: Use Sparse Assignment instead of merge_asof
        # merge_asof fills ALL rows, which causes performance freeze in preprocessing loops.
        # We only want to tag the specific nearest samples.
        
        # 1. Initialize empty label column
        # Use object type for strings, initialize with None or empty string?
        # Preprocessor expects to dropna(), so None is best, or empty string if we filter empty.
        # Let's use None/NaN for easy dropna().
        biopac_df['label'] = None 
        
        t_data = biopac_df['timestamp_ms'].values
        
        # 2. Assign nearest
        for _, row in label_df.iterrows():
            t_lbl = row['timestamp_ms']
            lbl_text = row['label']
            
            # Find closest sample index
            idx = np.searchsorted(t_data, t_lbl)
            
            nearest_idx = -1
            if idx == 0: nearest_idx = 0
            elif idx == len(t_data): nearest_idx = len(t_data) - 1
            else:
                d1 = abs(t_data[idx-1] - t_lbl)
                d2 = abs(t_data[idx] - t_lbl)
                nearest_idx = idx-1 if d1 < d2 else idx
                
            # Assign
            if 0 <= nearest_idx < len(biopac_df):
                # We need to assign to the 'label' column at this index
                # Use .iat for speed if possible, but we are modifying column
                # Get current val
                col_idx = biopac_df.columns.get_loc('label')
                current = biopac_df.iat[nearest_idx, col_idx]
                
                if pd.isna(current) or current is None:
                    biopac_df.iat[nearest_idx, col_idx] = lbl_text
                else:
                    biopac_df.iat[nearest_idx, col_idx] = f"{current}; {lbl_text}"
                    
        return biopac_df

    def load_acq(self, file_path: str, progress_callback: Optional[Callable[[str, float], None]] = None) -> Tuple[pd.DataFrame, float, int]:
        """
        Loads native .acq files using bioread.
        Handles mixed sample rates by resampling to max_fs.
        """
        try:
            import bioread
            from zoneinfo import ZoneInfo
        except ImportError:
            raise ImportError("bioread library is required for .acq files. `pip install bioread`")

        self.logger.info(f"Loading ACQ file: {file_path}")
        if progress_callback: progress_callback("Reading ACQ File...", 0.1)

        data = bioread.read_file(file_path)
        
        # 1. Identify Channels
        # Map generic names to found channels
        required_signals = {'ECG': ['ECG', 'EKG'], 'PPG': ['PPG', 'Pulse'], 'SKT': ['SKT', 'Temp'], 'EDA': ['EDA', 'GSR']}
        found_channels = {}
        
        target_channels = []
        max_fs = 0.0
        
        for chan in data.channels:
            c_name = chan.name.upper()
            fs = chan.samples_per_second
            max_fs = max(max_fs, fs)
            
            # Match against required
            matched_key = None
            for key, patterns in required_signals.items():
                for pat in patterns:
                    if pat in c_name:
                        matched_key = key
                        break
                if matched_key: break
            
            if matched_key and matched_key not in found_channels:
                found_channels[matched_key] = chan
                target_channels.append((matched_key, chan))
                self.logger.info(f"Matched {matched_key} to channel: {chan.name} ({fs} Hz)")
                
        if not found_channels:
            raise ValueError("No valid channels (ECG/PPG) found in .acq file.")
            
        # 2. Resample to Common Time Base (max_fs)
        if progress_callback: progress_callback("Resampling Channels...", 0.5)
        
        # Determine duration by looking at matched channels (use max duration)
        # Assuming all start at 0 relative in the file structure
        max_duration = max([c.time_index[-1] for _, c in target_channels])
        
        # Create common time vector
        # num_samples = duration * max_fs
        n_samples = int(max_duration * max_fs)
        common_time = np.linspace(0, max_duration, n_samples)
        
        df_dict = {}
        
        for key, chan in target_channels:
            # If fs matches max_fs closely, just use data (interpolated to strictly align)
            # Acq channels might be slightly diff length
            # We use np.interp for robust resampling
            resampled = np.interp(common_time, chan.time_index, chan.data)
            df_dict[key] = resampled
            
        df = pd.DataFrame(df_dict)
        
        # 3. Timestamps
        # data.earliest_marker_created_at is UTC (usually)
        # Convert to EST/EDT
        tz_ny = ZoneInfo("America/New_York")
        
        # Some acq files might check 'graph_created_at' or similar if markers are missing
        # Example provided: data.earliest_marker_created_at
        # If None, fallback?
        base_dt = data.earliest_marker_created_at
        
        if base_dt is None:
            # Try specific channel markers or fallback to now?
            # Or graph date
            # data.graph_header.created_at might exist? Bioread logic varies.
            # Let's assume user example holds true. If None, warn.
            self.logger.warning("No start timestamp found in ACQ. Using current time.")
            base_dt = datetime.now(tz_ny)
        else:
            # Ensure it is timezone aware
            if base_dt.tzinfo is None:
                base_dt = base_dt.replace(tzinfo=ZoneInfo("UTC")) # bioread usually returns UTC native
            
            # Convert
            base_dt = base_dt.astimezone(tz_ny)
            
        start_ts_ms = int(base_dt.timestamp() * 1000)
        
        # Add timestamp_ms column
        # common_time is 0-based seconds
        time_ms = start_ts_ms + (common_time * 1000).astype(int)
        df['timestamp_ms'] = time_ms
        
        return df, max_fs, start_ts_ms

    def load_and_stitch_acq(self, file_paths: list[str], progress_callback: Optional[Callable[[str, float], None]] = None) -> Tuple[pd.DataFrame, float, int]:
        """
        Loads multiple ACQ files and stitches them together, preserving time linearity.
        Inserts NaN gaps if files are not contiguous.
        """
        if not file_paths:
            raise ValueError("No file paths provided for stitching.")
            
        results = []
        total_files = len(file_paths)
        
        # 1. Load All Files
        for i, fp in enumerate(file_paths):
            print(f"DEBUG: Stitched Load {i+1}/{total_files}: {os.path.basename(fp)}")
            if progress_callback: 
                progress_callback(f"Loading file {i+1}/{total_files}: {os.path.basename(fp)}", 0.1 + (0.4 * (i/total_files)))
            
            # Use existing load_acq logic
            try:
                df, fs, start_ts = self.load_acq(fp)
                print(f"DEBUG: Finished loading {os.path.basename(fp)}. Rows: {len(df)}")
            except Exception as e:
                print(f"DEBUG: Failed to load {os.path.basename(fp)}: {e}")
                raise e

            results.append({
                'df': df,
                'fs': fs,
                'start': start_ts,
                'end': start_ts + (len(df) / fs * 1000.0), # Approximate end in ms
                'name': os.path.basename(fp)
            })
            
        # 2. Sort by Start Time
        results.sort(key=lambda x: x['start'])
        
        # 3. Stitch
        stitched_dfs = []
        current_fs = results[0]['fs']
        
        # Validate Sampling Rate Consistency
        for res in results:
            if abs(res['fs'] - current_fs) > 1.0:
                 self.logger.warning(f"Sampling rate mismatch: {res['name']} has {res['fs']}Hz, expected {current_fs}Hz. ")
                 # We could resample here, but for now assuming same protocol.
                 
        start_ts_global = results[0]['start']
        
        if progress_callback: progress_callback("Stitching and Gap Filling...", 0.6)
        
        last_end_ts = start_ts_global
        
        for i, res in enumerate(results):
            df = res['df']
            start = res['start']
            
            # Check Gap
            gap_ms = start - last_end_ts
            
            # Tolerance: If gap is massive negative (overlap), trim? 
            # If gap is small positive/negative (jitter), ignore.
            # If gap is significant positive (missing data), fill NaN.
            
            # 1 second threshold for "Gap"
            if gap_ms > 1000:
                print(f"DEBUG: Detected gap of {gap_ms:.1f}ms before {res['name']}")
                
                # Create NaN DataFrame
                gap_seconds = gap_ms / 1000.0
                n_gap_samples = int(gap_seconds * current_fs)
                
                # SAFETY: Prevent massive allocation if files are days apart
                # Limit to e.g., 6 hours @ 1000Hz = ~21M samples.
                if n_gap_samples > (6 * 3600 * current_fs):
                    raise ValueError(f"Gap between files is too large ({gap_seconds/3600:.1f} hours). Stitching limit is 6 hours to prevent memory crash.")

                if n_gap_samples > 0:
                    gap_df = pd.DataFrame(np.nan, index=range(n_gap_samples), columns=df.columns)
                    
                    # Create timestamps for gap
                    # last_end_ts is where previous file ended (roughly)
                    # Actually, better to construct time vector at the end or strictly per segment?
                    # Let's trust linear concatenation.
                    
                    # Fill 'timestamp_ms' for gap? 
                    # We will regenerate timestamp_ms for the whole merged df to be perfectly linear
                    # but we need columns to match for concat.
                    
                    stitched_dfs.append(gap_df)
            
            elif gap_ms < -1000:
                print(f"DEBUG: Overlap detected of {abs(gap_ms):.1f}ms. Trimming previous? Or just appending (double data)?")
                # User didn't specify overlap handling. 
                # Appending might cause time regression if we trust timestamps.
                # But here we are just concatenating rows. 
                # Ideally we want strictly increasing time.
                pass
                
            stitched_dfs.append(df)
            
            # Update last_end
            # Accurately: start + duration
            duration_ms = (len(df) / current_fs) * 1000.0
            last_end_ts = start + duration_ms
            
        # 4. Concatenate
        final_df = pd.concat(stitched_dfs, ignore_index=True)
        
        if progress_callback: progress_callback("Regenerating Timestamps...", 0.9)
        
        # 5. Regenerate complete Time Vector to ensure linearity
        # Start at global start, increment by 1/fs
        n_total = len(final_df)
        time_vec = start_ts_global + (np.arange(n_total) * (1000.0 / current_fs))
        final_df['timestamp_ms'] = time_vec.astype('int64')
        
        return final_df, current_fs, start_ts_global
